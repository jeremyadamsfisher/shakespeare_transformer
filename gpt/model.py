import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from torch.nn import functional as F


class MSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hs = self.config.n_embed // self.config.n_heads
        self.kqv = nn.Linear(self.config.n_embed, 3 * self.config.n_embed, bias=False)
        self.W = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.dropout = nn.Dropout(config.p_dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)) == 0
        self.register_buffer("mask", mask)

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
        assert C == self.config.n_embed
        # Attention needs to be parametric, so we begin with a learnable
        # linear transformation of the input. This is expressed as a single
        # matrix computation for efficiency
        x = self.kqv(x)
        assert x.shape == (B, T, 3 * C)
        # Break up the kqv computation into their constituents and split them
        # into orthoganol heads
        k, q, v = rearrange(x, "b t (s nh hs) -> s b nh t hs", s=3, hs=self.hs)
        if self.config.flash:
            # Use the efficient built-in attention module
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.p_dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Or compute it ourselves
            affinity_scores = einsum(q, k, "b nh ta hsa, b nh tb hsb -> b nh ta tb")
            assert affinity_scores.shape == (B, self.config.n_heads, T, T)
            # Softmax will quickly converge on producing one-hot vectors, whereas
            # we want each output token to be a mix of the input tokens. So we
            # normalize by the square root of the output dimensions.
            affinity_scores /= math.sqrt(self.hs)
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
            assert affinity.shape == (B, self.config.n_heads, T, T)
            # Consider the leftmost output token, which is the vector of dot products
            # of the first row of attention and all the value channel columns for all
            # tokens. Because all the logits are zero in the first row of attention
            # except for the first one, that output token is just the corresponding
            # value column. For the second output token, its the same matrix operation
            # but the attention can be split between the first and second value columns.
            # And so on.
            x = affinity @ v
        assert x.shape == (B, self.config.n_heads, T, self.hs)
        # Concatenate the results
        x = rearrange(x, "b nh t hs -> b t (nh hs)")
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


class Gpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.attention_blocks = nn.Sequential(
            *[GptBlock(config) for _ in range(config.n_layers)]
        )
        self.post_attention_norm = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.config = config

        if config.weight_tying:
            self.token_embedding.weight = self.lm_head.weight

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
