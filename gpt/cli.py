import typer

from gpt.utils import run_with_wanbd

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(large: bool = False, use_wandb: bool = False):
    from gpt.config import gpt_large, gpt_small
    from gpt.model import Gpt
    from gpt.train import train as train_

    config = gpt_large if large else gpt_small

    with run_with_wanbd(skip=use_wandb is False) as (m, pl_train_kwargs):
        model = train_(Gpt(config), config, pl_train_kwargs)
        m(model)  # log model, if wandb is enabled


if __name__ == "__main__":
    app()
