import math

import lightning.pytorch as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F


class LM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        self.log("trn_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("tst_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.one_cycle_scheduler is False:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [scheduler]

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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embed // config.n_heads
        self.key = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.dropout = nn.Dropout(config.p_dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)) == 0
        self.register_buffer("mask", mask)
        self.config = config

    def get_attention_mask(self, T):
        """Get an attention mask for a sequence of length T

        Examples:
            >>> Attention().get_attention_mask(2)
            torch.tensor([[0,1],
                          [0,0]])
        """
        return self.mask[:T, :T]

    def forward(self, x):
        B, T, C = x.shape
        assert (
            C
            == self.config.n_embed
            == self.key.in_features
            == self.query.in_features
            == self.value.in_features
        )
        # Attention needs to be parametric, so we begin with a learnable
        # linear transformation of the input
        k, q, v = self.key(x), self.query(x), self.value(x)
        # The query and key matrices cannot be multiplied directly; the query
        # matrix needs to be transposed such that all channel vectors are
        # dotted with one another
        affinity_scores = q @ rearrange(k, "b t c -> b c t")
        assert affinity_scores.shape == (B, T, T)
        # Softmax will quickly converge on producing one-hot vectors, whereas
        # we want each output token to be a mix of the input tokens. So we
        # normalize by the square root of the output dimensions.
        affinity_scores /= math.sqrt(self.head_size)
        # Recall that e^−∞ = 0. By setting the weights in the upper, right triangle
        # of the attention to −∞, the softmax allocates those weights to the past
        # and present tokens.
        affinity_scores = affinity_scores.masked_fill(
            self.get_attention_mask(T), -float("inf")
        )
        # Convert to a probability distribution
        affinity = F.softmax(affinity_scores, dim=-1)
        # Randomly drop some of the affinities to encourage regularization
        affinity = self.dropout(affinity)
        # Occasionally, dropouts produce NaNs, whereas we want 0s.
        # https://discuss.pytorch.org/t/getting-nans-from-dropout-layer/70693
        affinity = torch.nan_to_num(affinity, nan=0.0)
        assert affinity.shape == (B, T, T)
        # Consider the leftmost output token, which is the vector of dot products
        # of the first row of attention and all the value channel columns for all
        # tokens. Because all the logits are zero in the first row of attention
        # except for the first one, that output token is just the corresponding
        # value column. For the second output token, its the same matrix operation
        # but the attention can be split between the first and second value columns.
        # And so on.
        result = affinity @ v
        assert result.shape == (B, T, self.head_size)
        return result


class MSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Attention(config) for _ in range(config.n_heads)])
        self.W = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        B, T, C = x.shape
        # project the input logits into n orthogonal subspaces
        # within which attention is computed
        x = torch.stack([head(x) for head in self.heads], dim=0)
        # concatenate the attention-transformed subspaces
        x = rearrange(x, "h b t hs -> b t (h hs)")
        assert x.shape == (B, T, C)
        # reweight the attention-transformed subspaces, see section 3.3
        return self.W(x)


class GptBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_a = nn.LayerNorm(config.n_embed)
        self.msa = MSA(config)
        self.dropout_a = nn.Dropout(config.p_dropout)
        self.norm_b = nn.LayerNorm(config.n_embed)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
        )
        self.dropout_b = nn.Dropout(config.p_dropout)

    def forward(self, x):
        x = x + self.dropout_a(self.msa(self.norm_a(x)))
        x = x + self.dropout_b(self.ffn(self.norm_b(x)))
        return x


class Gpt(LM):
    def __init__(self, config):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.attention_blocks = nn.Sequential(
            *[GptBlock(config) for _ in range(config.n_layers)]
        )
        self.post_attention_norm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.config = config
        self.save_hyperparameters(config.dict())
        self.apply(self._init_weights)

    def forward(self, idxs):
        B, T = idxs.shape
        token_embeddings = self.token_embedding(idxs)
        assert token_embeddings.shape == (B, T, self.config.n_embed)
        pos_idxs = torch.arange(T, device=idxs.device)
        position_embedddings = self.position_embedding(pos_idxs)
        position_embedddings = repeat(position_embedddings, "t c -> b t c", b=B)
        assert position_embedddings.shape == (B, T, self.config.n_embed)
        x = token_embeddings + position_embedddings
        assert x.shape == (B, T, self.config.n_embed)
        x = self.attention_blocks(x)
        assert x.shape == (B, T, self.config.n_embed)
        x = self.post_attention_norm(x)
        assert x.shape == (B, T, self.config.n_embed)
        x = self.lm_head(x)
        assert x.shape == (B, T, self.config.vocab_size)
        return x
