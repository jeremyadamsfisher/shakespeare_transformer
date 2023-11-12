from typing import Optional

from pydantic import BaseModel, model_validator


class OneCycleLRConfig(BaseModel):
    pct_start: float = 0.3
    div_factor: float = 25
    final_div_factor: float = 1e4


class GptConfig(BaseModel):
    # architecture-specific
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int
    single_shot_attention: bool = False
    flash: bool = False
    batch_kqv: bool = False
    weight_tying: bool = False

    # training-specific
    batch_size: int
    n_epochs: int = 1
    p_dropout: float = 0.0
    lr: float = 1e-3
    test_train_split: float = 0.1
    one_cycle_scheduler: bool = False
    one_cycle_config: OneCycleLRConfig = OneCycleLRConfig()
    accumulate_grad_batches: int = 1

    # tokenization
    vocab_size: int
    tokenizer: Optional[str] = None  # No tokenizer should give a character tokenization

    @model_validator(mode="after")
    def valid(self):
        if self.batch_kqv is False and self.flash is True:
            raise ValueError
        return self


# See: https://arxiv.org/pdf/2005.14165.pdf table 2.1, pg. 8
gpt3_small = GptConfig(
    batch_size=4,
    accumulate_grad_batches=16,
    block_size=2048,
    lr=0.00025,
    n_embed=768,
    n_heads=12,
    n_layers=12,
    p_dropout=0.2,
    n_epochs=1,
    tokenizer="gpt2",
    vocab_size=50257,
    batch_kqv=True,
    flash=True,
)

gpt3_small_char = gpt3_small.model_copy()
gpt3_small_char.tokenizer = None
gpt3_small_char.vocab_size = 75


gpt3_small_char_one_cycle = gpt3_small_char.model_copy()
gpt3_small_char_one_cycle.one_cycle_scheduler = True
gpt3_small_char_one_cycle.lr = 2e-4  # See notes, v0.0.22


gpt3_small_char_one_cycle_v2 = gpt3_small_char_one_cycle.model_copy()
gpt3_small_char_one_cycle_v2.one_cycle_scheduler = True
gpt3_small_char_one_cycle_v2.lr = 1e-3  # See notes, v0.0.23
gpt3_small_char_one_cycle_v2.one_cycle_config = OneCycleLRConfig(
    pct_start=0.2,
    div_factor=10,
    final_div_factor=1,
)


# https://wandb.ai/jfisher40/gpt-shakespeare/runs/2a70mtrg/overview?workspace=user-jfisher40
gpt_mini_v0 = GptConfig(
    batch_size=128,
    block_size=256,
    learning_rate=0.001,
    n_embed=512,
    n_heads=8,
    n_layers=8,
    p_dropout=0.2,
    n_epochs=1,
    tokenizer=None,
    vocab_size=75,
)

# Using half precision, trying a bigger batch
gpt_mini_v1 = gpt_mini_v0.model_copy()
gpt_mini_v1.batch_size = 180


# baby gpt https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
gpt_baby = GptConfig(
    batch_size=64,
    block_size=256,
    learning_rate=1e-3,
    n_embed=384,
    n_heads=6,
    n_layers=6,
    p_dropout=0.2,
    n_epochs=1,
    tokenizer=None,
    vocab_size=75,
)
