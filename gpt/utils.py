from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytorch_lightning as L
from loguru import logger

import wandb
from gpt import PROJECT_ID, VERSION
from gpt.config import GptConfig


def get_run_name(load_from: Optional[str]):
    """Generate a run name for wandb. If load_from is provided, use the
    run name which is the parent directory of the checkpoint."""

    if load_from:
        return Path(load_from).parent.name
    else:
        return f"run-v{VERSION}-{uuid4()}"


@contextmanager
def run_manager(disable_wandb, load_from):
    """Return a context manager for running the model and determining
    the run name.

    Args:
        disable_wandb: whether to disable wandb
        load_from: path to a checkpoint to load from
    """

    name = get_run_name(load_from)
    ctx = (
        nullcontext
        if disable_wandb
        else lambda: wandb.init(project=PROJECT_ID, name=name)
    )
    with ctx():
        yield name


def summarize(model, config: GptConfig, dm: L.LightningDataModule):
    """Summarize a GPT model.

    Args:
        model: GPT model
        config: GPT config
        dm: GPT data module
    """
    n_params = sum(param.numel() for param in model.parameters())
    n_tokens = len(dm.X_trn) * config.block_size
    logger.info(f"num. parameters: {n_params:,d}")
    logger.info(f"num. tokens: {n_tokens:,d}")
    logger.info(
        f"tokens/parameters: {n_tokens/n_params:.1f} (chinchilla-optimal is 20/1)"
    )
    example, _ = next(iter(dm.train_dataloader()))
    first_example = example[0, :]
    first_example = dm.decode(first_example)[:100]
    logger.info(f"example batch (decoded): {first_example}")
    logger.info(f"config: {config}")
