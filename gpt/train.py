import os

import hydra
import pytorch_lightning as L
from loguru import logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from gpt.callbacks import LogGenerationPeriodically
from gpt.config import Config
from gpt.lightning_module import GptLightning
from gpt.utils import (
    check_for_repo_versioned_without_uncommited_changes,
    run_manager,
    summarize,
)
from gpt.wikipedia import WikipediaDataModule


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg: Config):
    """Train a GPT model."""
    if cfg.dirty is False:
        check_for_repo_versioned_without_uncommited_changes()

    with run_manager(cfg.disable_wandb, cfg.load_from) as name:
        dm = WikipediaDataModule(cfg.n_articles, cfg.model_config, profile=cfg.profile)
        model = GptLightning(cfg.model_config, compile=cfg.compile)

        if cfg.load_from is None:
            model.init_weights()

        dm.prepare_data()
        dm.setup("fit")

        # summarize(model, cfg, dm)

        logger_ = CSVLogger("./csv_logs") if cfg.disable_wandb else WandbLogger()
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            LogGenerationPeriodically(
                dm.decode, cfg.log_periodicity, None if cfg.disable_wandb else logger_
            ),
        ]

        if cfg.save_to:
            model_cb = ModelCheckpoint(
                dirpath=os.path.join(cfg.save_to, name),
                filename="{epoch}-{tst_loss:.2f}",
                every_n_train_steps=10_000,
                save_top_k=1,
                mode="min",
                monitor="tst_loss",
            )
            callbacks.append(model_cb)

        trainer = L.Trainer(
            max_epochs=cfg.model_config.n_epochs,
            callbacks=callbacks,
            logger=[logger_],
            val_check_interval=1000,
            accelerator="auto",
            profiler="simple" if cfg.profile else None,
            fast_dev_run=10 if cfg.profile else None,
            precision="bf16-mixed",
            accumulate_grad_batches=cfg.model_config.accumulate_grad_batches,
            default_root_dir=cfg.save_to,
        )

        if cfg.load_from:
            trainer.fit(model, dm, ckpt_path=cfg.load_from)
        else:
            trainer.fit(model, dm)

        return model


if __name__ == "__main__":
    train()
