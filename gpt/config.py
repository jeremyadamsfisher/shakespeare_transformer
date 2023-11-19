from dataclasses import dataclass
from typing import Optional



@dataclass
class OneCycleLRConfig:
    pct_start: float
    div_factor: float
    final_div_factor: float


@dataclass
class GptConfig:
    # architecture-specific
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int
    flash: bool
    weight_tying: bool

    # training-specific
    batch_size: int
    n_epochs: int
    p_dropout: float
    lr: float
    test_train_split: float
    one_cycle_scheduler: bool
    one_cycle_config: OneCycleLRConfig
    accumulate_grad_batches: int

    # tokenization
    vocab_size: int
    tokenizer: Optional[str]  # None should give a character tokenization


@dataclass
class DatasetConfig:
    dataset_uri: str
    tokenizer: Optional[str]
    block_size: int


@dataclass
class Config:
    log_periodicity: int
    profile: bool
    disable_wandb: bool
    load_from: str
    save_to: str
    dirty: bool
    model_config: GptConfig
    compile: bool
    data_config: "DatasetConfig"
