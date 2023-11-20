import multiprocessing as mp
from pathlib import Path

import pytorch_lightning as L
import torch
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_LOCAL_CACHE = "wikipedia_ds"


class ShiftedSequenceDataset:
    """A dataset that returns a block of tokens and a paired block of tokens
    shifted by one."""

    def __init__(self, config, ds):
        self.ds = ds
        self.config = config

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        tokens = self.ds[i]["tokens"]
        x, y = tokens[:-1], tokens[1:]
        assert len(x) == len(y) == self.config.model_config.block_size
        return x, y


class WikipediaDataModule(L.LightningDataModule):
    """Data module for wikipedia. Fairly generic and can should be able to be
    adapted for any huggingface dataset."""

    def __init__(self, config, n_workers=min((mp.cpu_count()-1, 16)), profile=False):
        super().__init__()
        self.config = config
        self.n_workers = 0 if profile else n_workers
        self.profile = profile

        assert self.config.data_config.tokenizer == self.config.model_config.tokenizer
        assert self.config.data_config.block_size == self.config.model_config.block_size

        if config.model_config.tokenizer is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        else:
            tokenizer = CharTokenizer()

        self.encode, self.decode = tokenizer.encode, tokenizer.decode

        if config.model_config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"please set vocab size to {tokenizer.vocab_size}")

    def prepare_data(self):
        if Path(WIKIPEDIA_LOCAL_CACHE).exists():
            logger.info("loading dataset from disk")
            return load_from_disk(WIKIPEDIA_LOCAL_CACHE)
        else:
            logger.info("loading dataset from google cloud")
            return load_from_disk(self.config.data_config.dataset_uri)

    def setup(self, stage=None):
        """Load dataset from cache directory. Re-loading from the disk is important
        here because each process will open its own memory mapped copy of the dataset,
        preventing resource locks."""

        if stage != "fit":
            return

        # Note that this should be an mmap if the dataset is already cached.
        # See: https://huggingface.co/docs/datasets/v2.14.5/en/use_with_pytorch#use-multiple-workers
        ds = self.prepare_data().with_format("torch", dtype=torch.long)

        self.X_trn = ShiftedSequenceDataset(self.config, ds["train"])
        self.X_tst = ShiftedSequenceDataset(self.config, ds["test"])

    def train_dataloader(self):
        return DataLoader(
            self.X_trn,
            shuffle=True,
            batch_size=self.config.model_config.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.X_tst,
            batch_size=self.config.model_config.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
        )
