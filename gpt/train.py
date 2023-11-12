import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytorch_lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torch.nn import functional as F

import wandb
from gpt import PROJECT_ID, VERSION
from gpt.config import GptConfig
from gpt.model import Gpt
from gpt.utils import get_run_name, run_manager


class GptLightning(L.LightningModule):
    def __init__(self, config, compile=True):
        super().__init__()
        self.config = config
        model = Gpt(config)
        if compile:
            model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.save_hyperparameters(config.dict())

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def step(self, batch):
        xb, yb = batch
        logits = self.forward(xb)
        B, T, C = logits.shape
        assert C == self.config.vocab_size
        return F.cross_entropy(
            rearrange(logits, "b t c -> (b t) c"),
            rearrange(yb, "b t -> (b t)"),
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("trn_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("tst_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.one_cycle_scheduler is False:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=self.config.lr,
                        total_steps=self.trainer.estimated_stepping_batches,
                        pct_start=self.config.one_cycle_config.pct_start,
                        div_factor=self.config.one_cycle_config.div_factor,
                        final_div_factor=self.config.one_cycle_config.final_div_factor,
                    ),
                    "interval": "step",
                    "frequency": 1,  # Update the LR every step
                    "monitor": "tst_loss",  # Not relevant for OneCycleLR
                    "strict": True,  # Doesn't need to be strict because the monitor is irrelevant
                },
            }

    @torch.no_grad()
    def generate(self, idxs=None, max_new_tokens: Optional[int] = None):
        """Generate a sequence of tokens from the model."""

        if max_new_tokens is None:
            max_new_tokens = self.config.block_size
        if idxs is None:
            idxs = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            # Our position embedding has a maximum length, so if the input is
            # longer than the block size, then crop it.
            idxs_cropped = idxs[:, -self.config.block_size :]
            assert idxs_cropped.shape[0] == 1
            assert idxs_cropped.shape[1] <= self.config.block_size
            # Get the model output
            logits = self(idxs_cropped)
            assert logits.shape[0] == 1
            assert idxs_cropped.shape[1] <= self.config.block_size
            assert logits.shape[2] == self.config.vocab_size
            # The model predicts logits for the probabilities for all the tokens,
            # i.e.: a shifted version of the input with the new token in the final
            # "time" position. We only need this last position.
            logits = logits[:, -1, :]
            assert logits.shape == (1, self.config.vocab_size)
            # Use these logits to create a probability distribution and
            # sample from it.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Finally, append it to the current sequence
            idxs = torch.cat([idxs, idx_next], dim=1)  # (B,T+1)
            # ...and repeat
        # strip the new line and remove singleton dimension
        idxs = idxs[0, 1:]
        return idxs

    @staticmethod
    def _init_weights(module):
        """Stolen from nanoGPT. TODO: understand this better."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self):
        self.apply(self._init_weights)


def train(
    model,
    config: GptConfig,
    dm: L.LightningDataModule,
    log_periodicity=500,
    profile=False,
    disable_wandb=True,
    load_from=None,
    save_to=None,
):
    """Train a GPT model.

    Args:
        model: GPT model
        config: GPT config
        dm: GPT data module
        log_periodicity: how often to log a generation
        profile: whether to profile the training
        disable_wandb: whether to disable wandb
        load_from: path to a checkpoint to load from
        save_to: path to save checkpoints to

    Returns:
        Trained GPT model
    """
    with run_manager(disable_wandb, load_from) as name:
        if load_from is None:
            model.init_weights()

        dm.prepare_data()
        dm.setup("fit")

        summarize(model, config, dm)

        logger_ = CSVLogger() if disable_wandb else WandbLogger()
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            LogGenerationPeriodically(
                dm.decode, log_periodicity, logger_ if disable_wandb is False else None
            ),
        ]

        if save_to:
            model_cb = ModelCheckpoint(
                dirpath=os.path.join(save_to, name),
                filename="{epoch}-{tst_loss:.2f}",
                every_n_train_steps=10_000,
                save_top_k=1,
                mode="min",
                monitor="tst_loss",
            )
            callbacks.append(model_cb)

        trainer = L.Trainer(
            max_epochs=config.n_epochs,
            callbacks=callbacks,
            logger=[logger_],
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
