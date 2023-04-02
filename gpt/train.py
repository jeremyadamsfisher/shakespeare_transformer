import lightning.pytorch as pl

from gpt.config import GptConfig
from gpt.data import ShakespeareDataModule


class LogGenerationPeriodically(pl.Callback):
    ...


def train(model, config: GptConfig):
    dm = ShakespeareDataModule(config, workers=-1)
    dm.setup()
    trainer = pl.Trainer(max_epochs=config.n_epochs)
    trainer.fit(model, dm)
    return model
