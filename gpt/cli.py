import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(config: str, log_periodicity: int = 100):
    # import here to avoid doing so for --help ingress
    from gpt.config import gpt_medium, gpt_small, gpt_small_one_cycle
    from gpt.model import Gpt
    from gpt.train import train as train_

    try:
        model_config = {
            # "large": gpt_large,
            "medium": gpt_medium,
            "small": gpt_small,
            "small_one_cycle": gpt_small_one_cycle,
        }[config.replace("-", "_").lower()]
    except KeyError:
        print(f"Unknown config: {config}")
        return

    model = Gpt(model_config)
    train_(model, model_config, log_periodicity)


@app.command()
def karpathy():
    from gpt.karpathy.train import train_karpathy

    train_karpathy()


if __name__ == "__main__":
    app()
