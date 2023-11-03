from setuptools import find_packages, setup

requirements = [
    "torch",
    "torchvision",
    "lightning",
    "einops",
    "typer",
    "rich",
    "shellingham",
    "wandb",
    "loguru",
    "datasets",
    "apache_beam",
    "mwparserfromhell",
    "pydantic",
    "Unidecode",
    "GitPython",
    "tokenizers",
]

setup(
    name="gpt",
    version="0.0.4",
    packages=find_packages(),
    install_requires=requirements,
    entry_points="""
        [console_scripts]
        train-gpt=gpt.cli:app
    """,
)
