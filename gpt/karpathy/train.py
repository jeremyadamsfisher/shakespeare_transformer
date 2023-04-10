import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from gpt.config import gpt_medium
from gpt.data import ShakespeareDataModule
from gpt.karpathy.model import GPTLanguageModel
from gpt.train import LogGenerationPeriodically


def train_karpathy():
    config = gpt_medium
    with wandb.init(project="gpt-shakespeare") as run:
        dm = ShakespeareDataModule(config, workers=-1)
        dm.setup()
        logger = WandbLogger()
        log_cb = LogGenerationPeriodically(dm.decode, 100, logger)
        trainer = pl.Trainer(
            max_epochs=config.n_epochs,
            callbacks=[log_cb],
            logger=[logger],
            val_check_interval=0.1,
            precision="16-mixed",
        )
        model = GPTLanguageModel()
        trainer.fit(model, dm)
