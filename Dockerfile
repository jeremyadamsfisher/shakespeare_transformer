FROM mambaorg/micromamba:jammy-cuda-11.8.0 AS base
WORKDIR /content
ADD env.cuda.yml env.yml ./
RUN micromamba env update ---name base --file env.yml
# # https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#using-a-lockfile
# RUN micromamba install --name base --yes --file conda-linux-64.lock \
#  && micromamba clean --all --yes

# Git not installed
ENV SHAKESPEARE_TRANSFORMER_IGNORE_GIT=True
ADD ./gpt ./gpt