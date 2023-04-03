from pydantic import BaseModel


class GptConfig(BaseModel):
    # architecture-specific
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int

    # dataset-specific
    vocab_size: int = 65

    # training-specific
    batch_size: int
    n_epochs: int = 1
    p_dropout: float = 0.0
    lr: float = 1e-3
    test_train_split: float = 0.1


gpt_small = GptConfig(
    batch_size=1024,
    block_size=32,
    n_embed=64,
    n_heads=4,
    n_layers=4,
    p_dropout=0.0,
    lr=1e-2,
)

gpt_large = GptConfig(
    batch_size=64, block_size=256, n_embed=384, n_heads=6, n_layers=6, p_dropout=0.2
)
