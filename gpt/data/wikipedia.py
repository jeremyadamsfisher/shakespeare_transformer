from datasets import load_dataset
from torch.utils.data import DataLoader

from . import CharDataModule, CharDataset


class WikipediaDataModule(CharDataModule):
    def __init__(self, config):
        super().__init__(config=config)

    def setup(self, stage=None):
        self.vocab_size = self.tokenizer.vocab_size
        ds = load_dataset("wikipedia", "20220301.en", split="train")
        dsx = ds.train_test_split(test_size=0.01)
        dsx["train"].set_format("torch", columns=["text"])
        dsx["test"].set_format("torch", columns=["text"])
        self.X_trn = CharDataset(self.config, dsx["train"], self.tokenizer)
        self.X_tst = CharDataset(self.config, dsx["test"], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.X_trn, shuffle=True, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.X_tst, batch_size=self.config.batch_size)
