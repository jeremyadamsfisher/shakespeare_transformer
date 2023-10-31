import multiprocessing
from pathlib import Path

import lightning.pytorch as pl
import requests
from torch.utils.data import DataLoader

from . import CharDataModule, CharDataset

DOWNLOAD_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespeareDataModule(CharDataModule):
    def __init__(self, config):
        super().__init__(config=config)
        self.data_fp = Path("./input.txt")
        self.config = config

    def setup(self, stage=None, tst_trn_split=0.1):
        if self.data_fp.exists():
            corpus = self.data_fp.read_text()
        else:
            corpus = requests.get(DOWNLOAD_URL).text
            with self.data_fp.open("w") as f:
                f.write(corpus)
        n = int(len(corpus) * tst_trn_split)
        corpus_trn, corpus_tst = corpus[n:], corpus[:n]
        self.encode, self.decode, self.vocab_size = get_encoder_decoder(corpus_trn)
        self.X_trn = CharDataset(self.encode(corpus_trn), self.config)
        self.X_tst = CharDataset(self.encode(corpus_tst), self.config)

    def train_dataloader(self):
        return DataLoader(
            self.X_trn,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.X_tst, batch_size=self.config.batch_size, num_workers=self.workers
        )
