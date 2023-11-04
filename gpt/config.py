from typing import Optional

from pydantic import BaseModel


class GptConfig(BaseModel):
    # architecture-specific
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int

    # training-specific
    batch_size: int
    n_epochs: int = 1
    p_dropout: float = 0.0
    lr: float = 1e-3
    test_train_split: float = 0.1
    one_cycle_scheduler: Optional[bool] = False

    # tokenization
    vocab_size: int
    tokenizer: Optional[str] = None  # No tokenizer should give a character tokenization


gpt_micro = GptConfig(
    batch_size=512,
    block_size=32,
    n_embed=64,
    n_heads=4,
    n_layers=4,
    p_dropout=0.0,
    lr=1e-2,
    n_epochs=4,
    tokenizer="gpt2",
    vocab_size=50257,
)

gpt_micro_one_cycle = gpt_micro.model_copy()
gpt_micro_one_cycle.one_cycle_scheduler = True

gpt_micro_char = gpt_micro.model_copy()
gpt_micro_char.tokenizer = None
gpt_micro_char.vocab_size = 75

gpt3_smaller = GptConfig(
    batch_size=1,
    block_size=2048,
    learning_rate=8e-4,
    n_embed=512,
    n_heads=8,
    n_layers=8,
    p_dropout=0.2,
    n_epochs=1,
    tokenizer="gpt2",
    vocab_size=50257,
)

# See: https://arxiv.org/pdf/2005.14165.pdf table 2.1
gpt3_small = GptConfig(
    batch_size=1,
    block_size=2048,
    learning_rate=6e-4,
    n_embed=768,
    n_heads=12,
    n_layers=12,
    p_dropout=0.2,
    n_epochs=1,
    tokenizer="gpt2",
    vocab_size=50257,
)
