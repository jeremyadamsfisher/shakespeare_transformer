from bisect import bisect_left
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


class ShiftedSequenceDataset:
    def __init__(self, config, ds):
        self.ds = ds
        self.config = config
        self.index = []
        for doc in tqdm(ds, unit="example", desc="Computing dataset index"):
            prev = self.index[-1] if self.index else -1
            n_blocks = len(doc["tokens"]) - (self.config.block_size + 1)
            self.index.append(prev + n_blocks)

    def __len__(self):
        return self.index[-1]

    def _get_idx_and_offset(self, i):
        ds_idx = bisect_left(self.index, i)
        if ds_idx == 0:
            offset = i
        else:
            offset = i - self.index[ds_idx - 1] - 1
        return ds_idx, offset

    def __getitem__(self, i):
        ds_idx, offset = self._get_idx_and_offset(i)
        tokens = self.ds[ds_idx]["tokens"]
        x = tokens[offset : offset + self.config.block_size]
        y = tokens[offset + 1 : offset + self.config.block_size + 1]
        x, y = map(torch.tensor, (x, y))
        return x, y


def tokenize_wikipedia_dataset(ds, tokenize: Callable[[str], Tensor], min_block_size):
    def wikipedia_batch_process(batch: Dict[str, Sequence]) -> Dict[str, Sequence]:
        tokens_batch = []
        for text in batch["text"]:
            tokens = tokenize(text)
            if len(tokens) < min_block_size:
                continue
            tokens_batch.append(tokens)
        return {"tokens": tokens_batch}

    return ds.map(
        wikipedia_batch_process,
        batched=True,
        remove_columns=["text"],
    )


class WikipediaDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
        load_dataset("jeremyf/tiny_wikipedia_en", split="train")

    @lru_cache
    def _setup(self):
        logger.info("tokenizing wikipedia")
        ds = load_dataset("jeremyf/tiny_wikipedia_en", split="train")
        ds = tokenize_wikipedia_dataset(
            ds,
            tokenize=self.encode,
            # Recall that we are predicting a shifted sequence
            min_block_size=self.config.block_size + 1,
        )
        dsx = ds.train_test_split(test_size=0.01)
        self.X_trn = ShiftedSequenceDataset(self.config, dsx["train"])
        self.X_tst = ShiftedSequenceDataset(self.config, dsx["test"])

    def setup(self, stage=None):
        self._setup()

    def train_dataloader(self):
        return DataLoader(self.X_trn, shuffle=True, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.X_tst, batch_size=self.config.batch_size)
