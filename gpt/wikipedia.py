import multiprocessing as mp
from bisect import bisect
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Sequence

import pytorch_lightning as L
import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_URI = "jeremyf/tiny_wikipedia_en"
WIKIPEDIA_LOCAL_CACHE = "wikipedia_ds"


class ShiftedSequenceDataset:
    def __init__(self, config, ds):
        self.ds = ds
        self.config = config
        self.compute_index()

    def compute_index(self):
        self.index = []
        for doc in tqdm(self.ds, unit="example", desc="Computing dataset index"):
            prev = self.index[-1] if self.index else 0
            n_blocks = len(doc["tokens"]) - self.config.block_size + 1
            # Note the we can't train on the last block because we
            # we don't have a label for the next token
            n_usable_blocks = n_blocks - 1
            self.index.append(prev + n_usable_blocks)

    def __len__(self):
        return self.index[-1]

    def _get_idx_and_offset(self, i):
        ds_idx = bisect(self.index, i)
        if ds_idx == 0:
            offset = i
        else:
            offset = i - self.index[ds_idx - 1]
        return ds_idx, offset

    def __getitem__(self, i):
        ds_idx, offset = self._get_idx_and_offset(i)
        tokens = self.ds[ds_idx]["tokens"]
        x = tokens[offset : offset + self.config.block_size]
        y = tokens[offset + 1 : offset + self.config.block_size + 1]

        if not (len(x) == len(y) == self.config.block_size):
            raise ValueError(
                f"i={i} ({ds_idx}, {offset}) does not produce a correctly sized tensor. "
                f"x: {len(x)}, y: {len(y)}"
            )

        return x, y


def tokenize_wikipedia_dataset(
    ds, tokenize: Callable[[str], Tensor], min_block_size,
):
    def wikipedia_batch_process(batch: Dict[str, Sequence]) -> Dict[str, Sequence]:
        tokens_batch = []
        for text in batch["text"]:
            tokens = tokenize(text)
            if min_block_size <= len(tokens):
                tokens_batch.append(tokens)
        return {"tokens": tokens_batch}

    ds = ds.map(
        wikipedia_batch_process,
        batched=True,
        remove_columns=["text"],
    )
    return ds


class WikipediaDataModule(L.LightningDataModule):
    def __init__(self, config, n_workers=mp.cpu_count(), profile=False):
        super().__init__()
        self.config = config
        self.n_workers = n_workers
        self.profile = profile

        if config.tokenizer is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        else:
            tokenizer = CharTokenizer()

        self.encode, self.decode = tokenizer.encode, tokenizer.decode

        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"please set vocab size to {tokenizer.vocab_size}")

    @lru_cache
    def prepare_data(self):
        """Save dataset to cache directory"""
        if Path(WIKIPEDIA_LOCAL_CACHE).exists():
            return
        ds = load_dataset(WIKIPEDIA_URI, split="train")
        ds = ds.select(range(1000))
        logger.info("tokenizing wikipedia")
        ds = tokenize_wikipedia_dataset(
            ds,
            tokenize=self.encode,
            # We need a source block that is at least one token bigger than the
            # context width of the model
            min_block_size=self.config.block_size + 1,
        )
        dsx = ds.train_test_split(test_size=0.01)
        dsx.save_to_disk(WIKIPEDIA_LOCAL_CACHE)

    @lru_cache
    def _setup(self):
        # Compute tokens and save to disk (may be called by lightning itself, but needs
        # to be called before `setup()``)
        self.prepare_data()

        # Memory-map: https://huggingface.co/docs/datasets/v2.14.5/en/use_with_pytorch#use-multiple-workers
        dsx = load_from_disk(WIKIPEDIA_LOCAL_CACHE).with_format("torch", dtype=torch.long)

        if self.profile:
            dsx = dsx.select(range(25))
        self.X_trn = ShiftedSequenceDataset(self.config, dsx["train"])
        self.X_tst = ShiftedSequenceDataset(self.config, dsx["test"])

    def setup(self, stage=None):
        self._setup()

    def train_dataloader(self):
        return DataLoader(
            self.X_trn,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.X_tst,
            batch_size=self.config.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )