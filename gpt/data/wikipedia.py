import lightning.pytorch as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from gpt.data import ShiftedSequenceDataset
from gpt.tokenizer import BpeTokenzizer


def tokenize_wikipedia_dataset(ds, tokenizer):
    def wikipedia_batch_process(batch):
        text = batch["text"]
        tokens = tokenizer.encode(text)
        return {"tokens": tokens}

    return ds.map(
        wikipedia_batch_process,
        batched=True,
        remove_columns=["id", "url", "title", "text"],
    )


class WikipediaDataModule(L.LightningDataModule):
    def __init__(self, config):
        self.config = config

    def setup(self, stage=None):
        import wandb

        self.vocab_size = self.tokenizer.vocab_size
        ds = load_dataset("wikipedia", "20220301.en", split="train")
        tokenizer = BpeTokenzizer.from_wandb_artifact(
            wandb.run, "shakespeare-tokenizer"
        )
        ds = tokenize_wikipedia_dataset(ds, tokenizer)
        dsx = ds.train_test_split(test_size=0.01)
        self.X_trn = ShiftedSequenceDataset(self.config, dsx["train"], self.tokenizer)
        self.X_tst = ShiftedSequenceDataset(self.config, dsx["test"], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.X_trn, shuffle=True, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.X_tst, batch_size=self.config.batch_size)
