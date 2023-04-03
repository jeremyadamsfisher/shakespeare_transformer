import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(large: bool = False):
    from gpt.config import gpt_large, gpt_small
    from gpt.model import Gpt
    from gpt.train import train as train_

    config = gpt_large if large else gpt_small
    train_(Gpt(config), config)


if __name__ == "__main__":
    app()
