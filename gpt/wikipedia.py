import json
import multiprocessing as mp
from bisect import bisect
from functools import lru_cache
from typing import Callable, Dict, Sequence

import pytorch_lightning as L
import torch
from datasets import load_dataset
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_URI = "jeremyf/tiny_wikipedia_en"


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
            offset = i - self.index[ds_idx-1]
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


def tokenize_wikipedia_dataset(ds, tokenize: Callable[[str], Tensor], min_block_size):
    def wikipedia_batch_process(batch: Dict[str, Sequence]) -> Dict[str, Sequence]:
        tokens_batch = []
        for text in batch["text"]:
            tokens = tokenize(text)
            if min_block_size <= len(tokens):
                tokens_batch.append(tokens)
        return {"tokens": tokens_batch}

    return ds.map(
        wikipedia_batch_process,
        batched=True,
        remove_columns=["text"],
    )


class WikipediaDataModule(L.LightningDataModule):
    def __init__(self, config, n_workers=mp.cpu_count()):
        super().__init__()
        self.config = config
        self.n_workers = n_workers

        if config.tokenizer is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        else:
            tokenizer = CharTokenizer()

        self.encode, self.decode = tokenizer.encode, tokenizer.decode

        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"please set vocab size to {tokenizer.vocab_size}")

    def prepare_data(self):
        """Save dataset to cache directory"""
        load_dataset(WIKIPEDIA_URI, split="train")

    @lru_cache
    def _setup(self):
        logger.info("tokenizing wikipedia")
        ds = load_dataset(WIKIPEDIA_URI, split="train")
        ds = tokenize_wikipedia_dataset(
            ds,
            tokenize=self.encode,
            # We need a source block that is at least one token bigger than the
            # context width of the model
            min_block_size=self.config.block_size + 1,
        )
        dsx = ds.train_test_split(test_size=0.01)
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
            collate_fn=wikipedia_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.X_tst,
            batch_size=self.config.batch_size,
            num_workers=self.n_workers,
            collate_fn=wikipedia_collator,
            pin_memory=True,
        )


def wikipedia_collator(examples):
    xs, ys = zip(*examples)
    xs, ys = map(torch.tensor, (xs, ys))
    return xs, ys