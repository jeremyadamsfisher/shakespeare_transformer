import tempfile
from contextlib import nullcontext

import pytorch_lightning as L
import torch
from loguru import logger

import wandb
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


def train(
    model,
    config: GptConfig,
    dm: L.LightningDataModule,
    log_periodicity=100,
    profile=False,
    silent=True,
):
    manager = (
        nullcontext
        if silent
        else lambda: wandb.init(project=PROJECT_ID, config={**config.dict()})
    )
    with manager() as run:
        dm.prepare_data()
        dm.setup()

        n_params = sum(param.numel() for param in model.parameters())
        n_tokens = len(dm.X_trn) * config.block_size
        logger.info(f"num. parameters: {n_params}")
        logger.info(f"num. tokens: {n_tokens}")
        logger.info(
            f"tokens/parameters: {n_tokens/n_params:.2f} (chinchilla-optimal is 20/1)"
        )

        example, _ = next(iter(dm.train_dataloader()))
        first_example = example[0, :]
        first_example = dm.decode(first_example)[:100]
        logger.info(f"example batch (decoded): {first_example}")

        wandb_logger = None if silent else L.loggers.WandbLogger()
        log_cb = LogGenerationPeriodically(dm.decode, log_periodicity, wandb_logger)
        lr_monitor = L.callbacks.LearningRateMonitor(logging_interval="step")
        trainer = L.Trainer(
            max_epochs=config.n_epochs,
            callbacks=[log_cb, lr_monitor],
            logger=[L.loggers.csv_logs.CSVLogger("./csv_logs")]
            if silent
            else [wandb_logger],
            val_check_interval=1000,
            accelerator="auto",
            profiler="simple" if profile else None,
            fast_dev_run=10 if profile else None,
            precision="bf16-mixed",
            accumulate_grad_batches=config.accumulate_grad_batches
        )
        trainer.fit(model, dm)
        if not silent:
            with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
                torch.save(model.state_dict(), f.name)
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(f.name)
                run.log_artifact(artifact)
        return model
