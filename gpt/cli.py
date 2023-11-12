import ast
import os
import re
from typing import Optional

import typer
from loguru import logger
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)


def check_for_repo_versioned_without_uncommited_changes():
    """If the current commit has unstaged/uncommited changes or lacks a version
    tag, throw an exception."""
    import git

    try:
        repo = git.Repo(search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        return False

    if not repo.head.is_valid():
        raise Exception("The current directory is not part of a Git repository.")

    for tag in repo.tags:
        if tag.commit.hexsha == repo.head.commit.hexsha:
            if re.match(r"v\d+\.\d+\.\d+", tag.name):
                break
    else:
        raise Exception("No version tag found in the current commit!")

    if repo.index.diff(None):
        raise Exception("Uncommitted changes!")


def get_model(config):
    from gpt import config as C

    return {
        "baby": C.gpt_baby,
        "small": C.gpt3_small,
        "small_char": C.gpt3_small_char,
        "small_char_a100": C.gpt3_small_char_a100,
        "small_char_one_cycle": C.gpt3_small_char_one_cycle,
        "small_char_one_cycle_v2": C.gpt3_small_char_one_cycle_v2,
        "mini_v0": C.gpt_mini_v0,
        "mini_v1": C.gpt_mini_v1,
    }[config.replace("-", "_").lower()]


save_to_help = "Checkpoint directory to save to. A UUID will be added. If unspecified, do not save the checkpoint"
load_from_help = "Checkpoint directory to load from. Please specify the UUID. If unspecified, do not load from a checkpoint"


@app.command()
def train(
    config: str,
    log_periodicity: int = 100,
    dirty: bool = False,
    disable_wandb: bool = False,
    profile: bool = False,
    save_to: Annotated[Optional[str], typer.Option(help=save_to_help)] = None,
    load_from: Annotated[Optional[str], typer.Option(help=load_from_help)] = None,
    compile: bool = False,
):
    from gpt.train import GptLightning
    from gpt.train import train as train_
    from gpt.wikipedia import WikipediaDataModule

    ignore_git = os.environ.get("SHAKESPEARE_TRANSFORMER_IGNORE_GIT", "False")
    ignore_git = ast.literal_eval(ignore_git)
    if not (dirty or ignore_git):
        check_for_repo_versioned_without_uncommited_changes()

    try:
        model_config = get_model(config)
    except KeyError:
        print(f"Unknown config: {config}")
        return

    logger.info("using config: {}", model_config)

    dm = WikipediaDataModule(model_config, profile=profile)
    model = GptLightning(model_config, compile=compile)

    save_to_env_var = os.environ.get("SHAKESPEARE_TRANSFORMER_SAVE_TO")
    if save_to is None and save_to_env_var:
        logger.info(
            "Using model checkpointing directory from SHAKESPEARE_TRANSFORMER_SAVE_TO: {}",
            save_to_env_var,
        )
        save_to = save_to_env_var

    if save_to and load_from:
        raise ValueError(
            "If loading from a checkpoint, you should save to the same directory!"
        )

    train_(
        model,
        model_config,
        dm,
        log_periodicity,
        profile,
        disable_wandb=disable_wandb,
        load_from=load_from,
        save_to=save_to,
    )


@app.command()
def find_lr(config: str, fname="./lr.png"):
    import pytorch_lightning as L

    from gpt.lightning_module import GptLightning
    from gpt.wikipedia import WikipediaDataModule

    try:
        model_config = get_model(config)
    except KeyError:
        print(f"Unknown config: {config}")
        return

    logger.info("initializing model and tuner")
    trainer = L.Trainer()
    tuner = L.tuner.Tuner(trainer)
    model = GptLightning(model_config, compile=False)
    dm = WikipediaDataModule(model_config, profile=False)
    dm.setup("fit")
    model.train_dataloader = dm.train_dataloader

    logger.info("finding the learning rate")
    lr_finder = tuner.lr_find(model)

    suggested_lr = lr_finder.suggestion()
    logger.info("suggested lr: {}", suggested_lr)

    fig = lr_finder.plot(suggest=True)
    fig.savefig(fname)


if __name__ == "__main__":
    app()
