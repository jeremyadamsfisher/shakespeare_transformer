from setuptools import find_packages, setup

requirements = ["torch", "torchvision", "lightning", "einops", "typer"]

setup(
    name="gpt",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
)
