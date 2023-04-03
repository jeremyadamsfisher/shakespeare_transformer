import tempfile

import lightning.pytorch as pl
import torch
from loguru import logger
from pytorch_lightning.loggers import WandbLogger

import wandb
from gpt.config import GptConfig
from gpt.data import ShakespeareDataModule


class LogGenerationPeriodically(pl.Callback):
    def __init__(self, decoder, log_periodicity=1000, wandb_logger=None):
        self.log_periodicity = log_periodicity
        self.decoder = decoder
        self.wandb_logger = wandb_logger

    def on_train_batch_start(self, _a, model, _b, batch_idx):
        if batch_idx % self.log_periodicity == 0:
            output = model.generate()
            output = self.decoder(output).replace("\n", " ")
            if self.wandb_logger:
                columns = ["generation"]
                data = [[output]]
                self.wandb_logger.log_text("trn/generation", columns=columns, data=data)
            logger.info("generation: {}", output)


def train(model, config: GptConfig, log_periodicity=1000):
    with wandb.init(project="gpt-shakespeare") as run:
        dm = ShakespeareDataModule(config, workers=-1)
        dm.setup()
        logger = WandbLogger()
        log_cb = LogGenerationPeriodically(dm.decode, log_periodicity, logger)
        trainer = pl.Trainer(
            max_epochs=config.n_epochs, callbacks=[log_cb], logger=[logger]
        )
        trainer.fit(model, dm)
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
            torch.save(model.state_dict(), f.name)
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(f.name)
            run.log_artifact(artifact)
        return model
