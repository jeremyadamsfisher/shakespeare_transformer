import string
from datasets import load_dataset
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        for char in string.printable:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, s: str):
        idxs = torch.tensor(
            [self.char2idx.get(char, self.char2idx["\n"]) for char in s],
            dtype=torch.long,
        )
        return idxs

    def decode(self, idxs: list[int]) -> str:
        return "".join(self.idx2char[int(i.item())] for i in idxs)

    @property
    def vocab_size(self):
        return len(self.char2idx)


class CharDataset(torch.utils.data.IterableDataset):
    def __init__(self, config, ds, split, tokenizer):
        self.X = ds[split].select_columns(["text"])
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self):
        for example in self.X:
            doc = example["text"]
            for idx in range(len(doc) - self.config.block_size - 1):
                x = doc[idx : idx + self.config.block_size]
                y = doc[idx + 1 : idx + self.config.block_size + 1]
                yield self.tokenizer.encode(x), self.tokenizer.encode(y)


class WikipediaDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = CharTokenizer()

    def encode(self, s):
        return self.tokenizer.encode(s)

    def decode(self, idxs):
        return self.tokenizer.decode(idxs)

    def setup(self, stage=None):
        self.vocab_size = self.tokenizer.vocab_size
        ds = load_dataset("wikipedia", "20220301.en", streaming=True)
        self.X_trn = CharDataset(self.config, ds, "train", self.tokenizer)
        # self.X_tst = CharDataset(self.config, ds, "test", self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.X_trn,
            batch_size=self.config.batch_size,
            num_workers=1,
            persistent_workers=True,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.X_tst,
    #         batch_size=self.config.batch_size,
    #         num_workers=1,
    #     )
