import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(large: bool = False, wandb: bool = False):
    from gpt.config import gpt_small
    from gpt.model import Gpt
    from gpt.train import train as train_

    if large:
        raise NotImplementedError
    if wandb:
        raise NotImplementedError

    config = gpt_small
    model = Gpt(config)
    train_(model, config)


if __name__ == "__main__":
    app()
