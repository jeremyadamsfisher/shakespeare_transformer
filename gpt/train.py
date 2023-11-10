from contextlib import nullcontext
from uuid import uuid4
import os
import pytorch_lightning as L
import wandb
from loguru import logger

from gpt import PROJECT_ID, VERSION
from gpt.config import GptConfig


def get_run_name():
    return f"run-v{VERSION}-{uuid4()}"


class LogGenerationPeriodically(L.Callback):
    def __init__(self, decoder, log_periodicity, wandb_logger=None):
        self.log_periodicity = log_periodicity
        self.decoder = decoder
        self.wandb_logger = wandb_logger

    def on_train_batch_start(self, trainer, model, _b, batch_idx):
        if batch_idx % self.log_periodicity == 0 and trainer.global_rank == 0:
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
    disable_wandb=True,
    load_from=None,
    save_to=None,
):
    manager = (
        nullcontext
        if disable_wandb
        else lambda: wandb.init(
            project=PROJECT_ID,
            config={**config.dict()},
            name=get_run_name(),
        )
    )
    with manager():
        if load_from is None:
            model.init_weights()

        dm.prepare_data()
        dm.setup()

        n_params = sum(param.numel() for param in model.parameters())
        n_tokens = len(dm.X_trn) * config.block_size
        logger.info(f"num. parameters: {n_params:,d}")
        logger.info(f"num. tokens: {n_tokens:,d}")
        logger.info(
            f"tokens/parameters: {n_tokens/n_params:.1f} (chinchilla-optimal is 20/1)"
        )

        example, _ = next(iter(dm.train_dataloader()))
        first_example = example[0, :]
        first_example = dm.decode(first_example)[:100]
        logger.info(f"example batch (decoded): {first_example}")

        wandb_logger = None if disable_wandb else L.loggers.WandbLogger()

        callbacks = [
            LogGenerationPeriodically(dm.decode, log_periodicity, wandb_logger),
            L.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

        if save_to:
            model_cb = L.callbacks.ModelCheckpoint(
                dir_path=os.path.join(save_to, get_run_name()),
                filename='shakespeare-transformer-{epoch}-{tst_loss:.2f}'
            )
            callbacks.append(model_cb)

        trainer = L.Trainer(
            max_epochs=config.n_epochs,
            callbacks=callbacks,
            logger=[L.loggers.csv_logs.CSVLogger("./csv_logs")]
            if disable_wandb
            else [wandb_logger],
            val_check_interval=1000,
            accelerator="auto",
            profiler="simple" if profile else None,
            fast_dev_run=10 if profile else None,
            precision="bf16-mixed",
            accumulate_grad_batches=config.accumulate_grad_batches,
            default_root_dir=save_to,
        )
        if load_from:
            trainer.fit(model, dm, ckpt_path=load_from)
        else:
            trainer.fit(model, dm)

        return model
