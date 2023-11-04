from functools import lru_cache, partial
from typing import Callable, Dict, Sequence

import pytorch_lightning as L
from datasets import load_dataset
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

from gpt.data import ShiftedSequenceDataset
from gpt.tokenizer import CharTokenizer


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
    def __init__(self, config, encode, decode):
        super().__init__()
        self.config = config
        self.encode = encode
        self.decode = decode

    @classmethod
    def with_char_tokenization(cls, config):
        tokenizer = CharTokenizer()
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"please set vocab size to {tokenizer.vocab_size}")
        return cls(config, encode=tokenizer.encode, decode=tokenizer.decode)

    @classmethod
    def with_bpe_tokenization(cls, config):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"please set vocab size to {tokenizer.vocab_size}")
        return cls(
            config,
            encode=partial(tokenizer.encode, return_tensors="pt"),
            decode=tokenizer.decode,
        )

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
