import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
from uuid import uuid4

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
    with (
        nullcontext
        if disable_wandb
        else lambda: wandb.init(project=PROJECT_ID, name=name)
    ):
        yield name


def summarize(model, config: GptConfig, dm: L.LightningDataModule):
    """Summarize a GPT model.

    Args:
        model: GPT model
        config: GPT config
        dm: GPT data module
    """
    example, _ = next(iter(dm.train_dataloader()))
    first_example = example[0, :]
    first_example = dm.decode(first_example)[:100]
    logger.info(f"example batch (decoded): {first_example}")
    logger.info(f"model: {model}")
    logger.info(f"config: {config}")
