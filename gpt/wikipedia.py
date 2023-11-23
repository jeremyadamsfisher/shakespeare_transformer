import shutil
import multiprocessing as mp
from pathlib import Path
from subprocess import check_call

import pytorch_lightning as L
import torch
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_LOCAL_CACHE = "./wikipedia_ds/"


class ShiftedSequenceDataset(torch.utils.data.Dataset):
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

    def __init__(self, config, n_workers=None, profile=False):
        super().__init__()
        self.config = config
        self.profile = profile

        if n_workers is None:
            if profile:
                self.n_workers = 0
            elif config.distributed:
                self.n_workers = 1
            else:
                self.n_workers = min((mp.cpu_count() - 1, 16))

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
            logger.info("dataset already exists on disk")
            return
        
        logger.info("downloading dataset")
        cmd = (
            "gsutil",
            "-m",
            "cp",
            "-r",
            self.config.data_config.dataset_uri,
            ".",
        )
        logger.info("running command: {}", " ".join(cmd))
        check_call(cmd)
        logger.info("finished downloading dataset")

        logger.info("moving dataset to local cache")
        dir_name = Path(self.config.data_config.dataset_uri).name
        shutil.move(
            dir_name,
            WIKIPEDIA_LOCAL_CACHE
        )
        logger.info("finished moving dataset to local cache")


    def setup(self, stage=None):
        """Load dataset from cache directory. Re-loading from the disk is important
        here because each process will open its own memory mapped copy of the dataset,
        preventing resource locks."""

        if stage != "fit":
            logger.info("skipping setup for stage: {stage} (nothing to do)")
            return

        # Note that this should be an mmap if the dataset is already cached.
        # See: https://huggingface.co/docs/datasets/v2.14.5/en/use_with_pytorch#use-multiple-workers
        ds = load_from_disk(WIKIPEDIA_LOCAL_CACHE)
        ds = ds.with_format("torch", dtype=torch.long)

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
