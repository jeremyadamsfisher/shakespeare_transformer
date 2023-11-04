import tempfile

import pytorch_lightning as L
import torch
import wandb
from loguru import logger

from gpt import PROJECT_ID
from gpt.config import GptConfig


class LogGenerationPeriodically(L.Callback):
    def __init__(self, decoder, log_periodicity, wandb_logger=None):
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


def train(model, config: GptConfig, dm: L.LightningDataModule, log_periodicity=100):
    with wandb.init(project=PROJECT_ID, config={**config.dict()}) as run:
        dm.setup()
        logger = L.loggers.WandbLogger()
        log_cb = LogGenerationPeriodically(dm.decode, log_periodicity, logger)
        trainer = L.Trainer(
            max_epochs=config.n_epochs,
            callbacks=[log_cb, L.callbacks.EarlyStopping("tst_loss")],
            logger=[logger],
            # val_check_interval=1000,
            precision="16-mixed",
            accelerator="auto",
        )
        trainer.fit(model, dm)
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
            torch.save(model.state_dict(), f.name)
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(f.name)
            run.log_artifact(artifact)
        return model
