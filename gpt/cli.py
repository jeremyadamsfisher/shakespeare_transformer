import typer

from gpt.config import gpt_small
from gpt.model import Gpt
from gpt.train import train as train_

app = typer.Typer()


@app.command()
def train(large: bool = False, wandb: bool = False):
    if large:
        raise NotImplementedError
    if wandb:
        raise NotImplementedError
    model = Gpt(gpt_small)
    train_(model)


if __name__ == "__main__":
    app()
