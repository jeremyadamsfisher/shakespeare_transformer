FROM mambaorg/micromamba:jammy-cuda-11.8.0 AS base
USER root
RUN apt update && apt install -y git build-essential
ADD conda-linux-64.lock .
# https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#using-a-lockfile
RUN micromamba install --name base --yes --file conda-linux-64.lock \
 && micromamba clean --all --yes
WORKDIR /app
ENV SHAKESPEARE_TRANSFORMER_IGNORE_GIT=True
ADD ./gpt ./gpt