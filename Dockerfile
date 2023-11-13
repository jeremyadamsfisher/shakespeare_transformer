FROM mambaorg/micromamba:jammy-cuda-11.8.0 AS base
WORKDIR /content
ADD requirements.txt .
RUN micromamba install -y --name base -c defaults python=3.8
RUN micromamba run --name base python -m pip install -r requirements.txt
ADD ./gpt ./gpt