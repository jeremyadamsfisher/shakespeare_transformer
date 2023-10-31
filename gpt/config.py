from typing import Optional

from pydantic import BaseModel


class GptConfig(BaseModel):
    # architecture-specific
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int

    # dataset-specific
    vocab_size: int = 75

    # training-specific
    batch_size: int
    n_epochs: int = 1
    p_dropout: float = 0.0
    lr: float = 1e-3
    test_train_split: float = 0.1
    one_cycle_scheduler: Optional[bool] = False


gpt_small = GptConfig(
    batch_size=512,
    block_size=32,
    n_embed=64,
    n_heads=4,
    n_layers=4,
    p_dropout=0.0,
    lr=1e-2,
    n_epochs=4,
)

gpt_small_one_cycle = gpt_small.copy()
gpt_small_one_cycle.one_cycle_scheduler = True

gpt_medium = GptConfig(
    batch_size=256,
    block_size=128,
    n_embed=384,
    n_heads=6,
    n_layers=6,
    p_dropout=0.2,
    lr=3e-4,
    n_epochs=10,
)

gpt_large = GptConfig(
    batch_size=64,
    block_size=256,
    learning_rate=3e-4,
    n_embed=384,
    n_heads=6,
    n_layers=6,
    p_dropout=0.2,
    n_epochs=10,
)

gpt_larger = GptConfig(
    batch_size=128,
    block_size=256,
    learning_rate=3e-4,
    n_embed=512,
    n_heads=8,
    n_layers=10,
    p_dropout=0.2,
    n_epochs=1,
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
)