import gzip
import json
import multiprocessing as mp
import os
from bisect import bisect
from pathlib import Path
from typing import Callable, Dict, Sequence

import hydra
import pytorch_lightning as L
import torch
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_URI = "wikipedia"
WIKIPEDIA_LOCAL_CACHE = "wikipedia_ds"


class ShiftedSequenceDataset:
    """A dataset that returns a block of tokens and a paired block of tokens
    shifted by one."""

    def __init__(self, config, ds, index_fp):
        self.ds = ds
        self.config = config
        if index_fp.exists():
            logger.info("loading index from disk")
            with gzip.open(index_fp) as f:
                self.index = json.load(f)
        else:
            self.compute_index()
            logger.info("dumping index to disk")
            with gzip.open(index_fp, "wt") as f:
                json.dump(self.index, f)

    def compute_index(self):
        """Compute the index of the dataset. The underlying dataset is a huggingface
        dataset of tokenized wikipedia articles. We want to return non-overlapping
        blocks of tokens, so we need to determine how many blocks are in each article
        to index into them efficiently.

        For example, imagine that our block size is 99 tokens. Then we if we have
        a dataset of 3 articles, and the first article has 100 tokens, the second
        has 200 tokens, and the third has 300 tokens, we have 6 useable blocks. Our
        dataset has a length of 6, and the index is [1, 3, 6]. If the dataloader
        requests index 3, we know that the 4th block is in the 3rd article by looking
        it up through binary search. We then know that the offset is 4 - 3 = 1, and
        we can return the tokens from the 1st block of the 3rd article."""

        self.index = []
        for doc in tqdm(self.ds, unit="example", desc="Computing dataset index"):
            prev = self.index[-1] if self.index else 0
            n_blocks = len(doc["tokens"]) // (self.config.block_size + 1)
            self.index.append(prev + n_blocks)

    def __len__(self):
        """Return the number of blocks in the dataset. Since the index is the
        cumulative sum of the number of blocks in each article, the last element
        of the index is the total number of blocks in the dataset."""
        return self.index[-1]

    def _get_idx_and_offset(self, i):
        """Return the index of the article and the offset into the article
        for the ith block."""
        ds_idx = bisect(self.index, i)
        if ds_idx == 0:
            offset = i
        else:
            offset = i - self.index[ds_idx - 1]
        return ds_idx, offset * self.config.block_size

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
    ds,
    tokenize: Callable[[str], Tensor],
    min_block_size,
):
    """Tokenize a dataset of wikipedia articles. We need to tokenize the articles
    before training because we need to know how many tokens are in each article
    to index into them."""

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
        num_proc=mp.cpu_count() - 1,
    )


class WikipediaDataModule(L.LightningDataModule):
    """Data module for wikipedia. Fairly generic and can should be able to be
    adapted for any huggingface dataset."""

    def __init__(self, n_articles, config, n_workers=mp.cpu_count(), profile=False):
        super().__init__()
        self.config = config
        self.n_workers = 0 if profile else n_workers
        self.profile = profile
        self.batch_size = config.batch_size
        self.n_articles = n_articles

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
        self.fp = os.path.join(hydra.utils.get_original_cwd(), WIKIPEDIA_LOCAL_CACHE)
        if Path(self.fp).exists():
            return

        if self.n_articles:
            ds_full = load_dataset(
                WIKIPEDIA_URI, "20220301.en", split="train", streaming=True
            )
            texts = []
            iter_ds = iter(ds_full)
            for _ in trange(self.n_articles, desc="Downloading wikipedia"):
                row = next(iter_ds)
                texts.append(row["text"])
            ds = Dataset.from_dict({"text": texts})
        else:
            ds = load_dataset(WIKIPEDIA_URI, "20220301.en", split="train")
            ds = ds.select_columns(["text"])

        logger.info("tokenizing wikipedia")
        ds = tokenize_wikipedia_dataset(
            ds,
            tokenize=self.encode,
            # We need a source block that is at least one token bigger than the
            # context width of the model
            min_block_size=self.config.block_size + 1,
        )
        dsx = ds.train_test_split(test_size=0.0025)
        dsx.save_to_disk(self.fp)

    def setup(self, stage=None):
        """Load dataset from cache directory. Re-loading from the disk is important
        here because each process with open its own memory mapped copy of the dataset,
        preventing resource locks."""

        if stage != "fit":
            return

        # Idempotent and neccesary to run beforehand, so no harm in programming defensively here
        self.prepare_data()

        # Memory-map: https://huggingface.co/docs/datasets/v2.14.5/en/use_with_pytorch#use-multiple-workers
        dsx = load_from_disk(self.fp).with_format("torch", dtype=torch.long)

        self.X_trn = ShiftedSequenceDataset(
            self.config,
            dsx["train"],
            Path(self.fp) / "wikipedia-index-trn.json.gz",
        )
        self.X_tst = ShiftedSequenceDataset(
            self.config,
            dsx["test"],
            Path(self.fp) / "wikipedia-index-tst.json.gz",
        )

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
