from typing import Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from gpt.config import GptConfig
from gpt.model import Gpt


class GptLightning(L.LightningModule):
    """Training and inference layer on top of the model itself. Designed to remove
    non-architecture code from the model itself. PyTorchLightning is a bit of a
    grab-bag of features, so this class is a bit of a grab-bag of features too."""

    def __init__(self, config: GptConfig, compile: bool = True):
        super().__init__()
        self.config = config
        model = Gpt(config)
        if compile:
            model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.save_hyperparameters(config._content)

    def step(self, batch):
        xb, yb = batch
        logits = self.model.forward(xb)
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

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        