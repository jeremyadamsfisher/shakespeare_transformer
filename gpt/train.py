import lightning.pytorch as pl
from loguru import logger

from gpt.config import GptConfig
from gpt.data import ShakespeareDataModule


class LogGenerationPeriodically(pl.Callback):
    def __init__(self, decoder, log_periodicity=1000):
        self.log_periodicity = log_periodicity
        self.decoder = decoder

    def on_train_batch_start(self, _a, model, _b, batch_idx):
        if batch_idx % self.log_periodicity == 0:
            output = model.generate()
            output = self.decoder(output).replace("\n", " ")
            msg = "TRN batch {:05}, generation: {}".format(batch_idx, output)
            logger.info(msg)


def train(model, config: GptConfig, log_periodicity=1000, pl_train_kwargs=None):
    if pl_train_kwargs is None:
        pl_train_kwargs = {}
    dm = ShakespeareDataModule(config, workers=-1)
    dm.setup()
    log_cb = LogGenerationPeriodically(dm.decode, log_periodicity)
    trainer = pl.Trainer(
        max_epochs=config.n_epochs, callbacks=[log_cb], **pl_train_kwargs
    )
    trainer.fit(model, dm)
    return model
