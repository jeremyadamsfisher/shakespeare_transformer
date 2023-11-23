from typing import Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch.nn import functional as F

from gpt.config import GptConfig
from gpt.model import Gpt
from gpt.utils import restore_config


class GptLightning(L.LightningModule):
    """Training and inference layer on top of the model itself. Designed to remove
    non-architecture code from the model itself. PyTorchLightning is a bit of a
    grab-bag of features, so this class is a bit of a grab-bag of features too."""

    def __init__(self, config: Union[dict, GptConfig]):
        super().__init__()

        if isinstance(config, dict):
            # This happens when loading from a checkpoint
            config = GptConfig(**restore_config(config))

        self.config = config
        model = Gpt(config)
        self.model = model

        try:
            self.save_hyperparameters(config._content)
        except AttributeError:
            logger.warning(
                "Could not save hyperparameters. "
                "Probably just restoring and the config is a bit mangled."
            )

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

    def forward(self, x):
        return self.model.forward(x)

    @torch.no_grad()
    def generate(self, idxs=None, max_new_tokens: Optional[int] = None):
        """Generate a sequence of tokens from the model.

        Needs to be here (as opposed to the underlying model) to know
        to use the model's device.

        Args:
            idxs: the initial sequence of tokens to start from
            max_new_tokens: the maximum number of tokens to generate

        Returns:
            The generated sequence of tokens"""

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
