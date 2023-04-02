import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class LM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def step(self, batch):
        xb, yb = batch
        logits = self.forward(xb)
        return F.cross_entropy(
            rearrange(logits, "b t c -> b (t c)"),
            rearrange(yb, "b t -> (b t)"),
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("trn_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("tst_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def generate(self, max_new_tokens: int = 50):
        idxs = torch.zeros(
            shape=(1, 1, self.config.n_embed), dtype=torch.long, device=self.device
        )
        for _ in range(max_new_tokens):
            # Our language model expects a fixed size input, so if the input is
            # longer than the block size, then crop it.
            idxs_cropped = idxs[:, -self.config.block_size :]  # (B,T)
            # Get the model output
            logits = self(idxs_cropped)  # (B,T,C)
            # The model predicts logits for the probabilities for all the tokens,
            # i.e.: a shifted version of the input with the new token in the final
            # "time" position. We only need this last position.
            logits = logits[:, -1, :]  # (B,C)
            # Use these logits to create a probability distribution and
            # sample from it.
            probs = F.softmax(logits, dim=-1)  # (B)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # Finally, append it to the current sequence
            idxs = torch.cat([idxs, idx_next], dim=1)  # (B,T+1)
            # ...and repeat
        # strip the new line and remove singleton dimension
        idxs = idxs[..., 1:].squeeze()
        return idxs


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads
        self.key = nn.Linear(self, config.n_embed, head_size)
        self.query = nn.Linear(self, config.n_embed, head_size)
        self.value = nn.Linear(self, config.n_embed, head_size)
        self.dropout = nn.Dropout(config.p_dropout)

    def get_attention_mask(self, T):
        """Get an attention mask for a sequence of length T

        Examples:
            >>> Attention().get_attention_mask(2)
            torch.tensor([[0,1],
                          [0,0]])
        """
        return torch.tril(torch.ones((T, T))) == 0

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
        affinity_scores = k @ rearrange(q, "b t c -> b c t")
        assert affinity_scores.shape == (B, T, T)
        # Softmax will quickly converge on producing one-hot vectors, whereas
        # we want each output token to be a mix of the input tokens. So we
        # normalize by the square root of the output dimensions.
        affinity_scores /= math.sqrt(C)
        # Recall that e^−∞ = 0. By setting the weights in the upper, left triangle
        # of the attention to −∞, the softmax allocates those weights to the past
        # and present tokens.
        affinity_scores.masked_fill(self.get_attention_mask(T), -float("inf"))
        # Randomly drop some of the affinities to encourage regularization
        affinity_scores = self.dropout(affinity_scores)
        # Convert to a probability distribution
        affinity = F.softmax(affinity_scores, dim=-1)
        assert affinity.shape == (B, T, T) and (affinity.sum(dim=-1) == 1).all()
        # Consider the leftmost output token, which is the vector of dot products
        # of the first row of attention and all the value channel columns for all
        # tokens. Because all the logits are zero in the first row of attention
        # except for the first one, that output token is just the corresponding
        # value column. For the second output token, its the same matrix operation
        # but the attention can be split between the first and second value columns.
        # And so on.
        return affinity @ v


class MSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(config) for _ in range(self.config.n_heads)]
        )
        self.W = nn.Linear(config.n_embed, config.n_embed)

    def foward(self, x):
        # project the input logits into n orthogonal subspaces
        # within which attention is computed
        x = torch.stack([head(x) for head in self.heads])
        # concatenate the attention-transformed subspaces
        x = rearrange(x, "b t h d -> b t (h d)")
        # reweight the attention-transformed subspaces, see section 3.3
        return self.W(x)


class GptBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_a = nn.LayerNorm(self.config.n_embed)
        self.dropout_a = nn.Dropout(config.p_dropout)
        self.msa = MSA(config)
        self.norm_b = nn.LayerNorm(self.config.n_embed)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.p_dropout),
        )
        self.dropout_b = nn.Dropout(config.p_dropout)

    def forward(self, x):
        x = x + self.msa(self.norm_a(x))
        x = self.dropout_a(x)
        x = x + self.ffn(self.norm_b(x))
        x = self.dropout_b(x)
        return x


class Gpt(LM):
    def __init__(self, config):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(
            config.context_window_size, config.n_embed
        )
        self.attention_blocks = nn.Sequential(
            *[GptBlock(config) for _ in range(config.n_layers)]
        )
        self.post_attention_norm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, idxs):
        B, T = idxs.shape
        token_embeddings = self.token_embedding(idxs)
        assert token_embeddings.shape == (B, T, self.config.n_embed)
        position_embedddings = self.position_embedding(idxs)
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
