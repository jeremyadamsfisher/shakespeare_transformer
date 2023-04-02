import tempfile
from contextlib import contextmanager

import wandb


@contextmanager
def run_with_wanbd(skip: bool):
    """run wandb conditionally"""
    if skip:
        yield lambda _: None, {}  # train model, don't do anything with the artifacts
        return

    import torch
    from pytorch_lightning.loggers import WandbLogger

    run = wandb.init(project="gpt-shakespeare")
    model = None

    def set_model(model_):
        nonlocal model
        model = model_

    yield set_model, {"logger": WandbLogger()}

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(model.state_dict(), f.name)
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f.name)
        run.log_artifact(artifact)
    wandb.finish()
