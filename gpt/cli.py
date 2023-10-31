import typer

app = typer.Typer()


@app.command()
def train(config: str, log_periodicity: int = 100):
    # import here to avoid doing so for --help ingress
    from gpt.config import gpt_micro, gpt_micro_one_cycle, gpt3_small
    from gpt.data.wikipedia import WikipediaDataModule
    from gpt.model import Gpt
    from gpt.train import train as train_

    try:
        model_config = {
            "micro": gpt_micro,
            "micro_one_cycle": gpt_micro_one_cycle,
            "gpt3_small": gpt3_small,
        }[config.replace("-", "_").lower()]
    except KeyError:
        print(f"Unknown config: {config}")
        return

    dm = WikipediaDataModule(config)

    model = Gpt(model_config)
    train_(model, model_config, dm, log_periodicity)


if __name__ == "__main__":
    app()
