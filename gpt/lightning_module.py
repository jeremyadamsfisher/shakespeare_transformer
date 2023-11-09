import pytorch_lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from gpt.model import Gpt


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
    def generate(self, idxs=None, max_new_tokens: int = 50):
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