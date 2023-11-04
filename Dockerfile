FROM mambaorg/micromamba:jammy-cuda-11.8.0
ADD conda-linux-64.lock .
# https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#using-a-lockfile
RUN micromamba install --name base --yes --file conda-linux-64.lock \
 && micromamba clean --all --yes
WORKDIR /app
ADD gpt ./gpt