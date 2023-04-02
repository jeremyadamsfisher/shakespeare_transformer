import multiprocessing
from pathlib import Path

import lightning.pytorch as pl
import requests
import torch
from torch.utils.data import DataLoader

DOWNLOAD_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_encoder_decoder(trn_corpus):
    chars = sorted(set(trn_corpus))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}

    def encode(s):
        return torch.tensor([char2idx[char] for char in s], dtype=torch.long)

    def decode(idxs):
        return "".join(idx2char[int(idx)] for idx in idxs)

    return encode, decode, vocab_size


class CharDataset:
    def __init__(self, corpus, config):
        self.corpus = corpus
        self.config = config

    def __len__(self):
        return len(self.corpus) - self.config.block_size

    def __getitem__(self, idx):
        x = self.corpus[idx : idx + self.config.block_size]
        y = self.corpus[idx + 1 : idx + self.config.block_size + 1]
        return x, y


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        workers=1,
    ):
        super().__init__()
        self.data_fp = Path("./input.txt")
        self.config = config
        self.workers = workers if workers > 0 else multiprocessing.cpu_count()

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
        return DataLoader(self.X_trn, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.X_tst, batch_size=self.config.batch_size)
