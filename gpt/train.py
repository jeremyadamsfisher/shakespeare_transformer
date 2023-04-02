import lightning.pytorch as pl

from gpt.config import GptConfig
from gpt.data import ShakespeareDataModule


def train(model, config: GptConfig):
    dm = ShakespeareDataModule(workers=-1)
    dm.setup()
    trainer = pl.Trainer(max_epochs=config.n_epochs)
    trainer.fit(model, dm)
    return model
