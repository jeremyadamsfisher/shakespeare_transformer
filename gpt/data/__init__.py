from random import randint

import lightning.pytorch as pl
from unidecode import unidecode

from gpt.tokenizer import CharTokenizer


class CharDataset:
    def __init__(self, config, ds, tokenizer):
        self.ds = ds
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """Get a random chunk"""
        example = self.ds[idx]
        doc = example["text"]
        doc = unidecode(doc).lower()

        if len(doc) < self.config.block_size + 1:
            # Skip anything smaller than the context window
            return self[randint(0, len(self)) - 1]
        else:
            idx = randint(0, len(doc) - self.config.block_size - 1)
            x = doc[idx : idx + self.config.block_size]
            y = doc[idx + 1 : idx + self.config.block_size + 1]

        return self.tokenizer.encode(x), self.tokenizer.encode(y)


class CharDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = CharTokenizer()
        self.config = config

    def encode(self, s):
        return self.tokenizer.encode(s)

    def decode(self, idxs):
        return self.tokenizer.decode(idxs)
