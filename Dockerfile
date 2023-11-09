FROM mambaorg/micromamba:jammy-cuda-11.8.0
USER root
RUN apt update && apt install -y git build-essential
ADD conda-linux-64.lock .
# https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html#using-a-lockfile
RUN micromamba install --name base --yes --file conda-linux-64.lock \
 && micromamba clean --all --yes
ADD docker-init.sh /docker-init.sh
RUN chmod +x /docker-init.sh
RUN echo 'bash /docker-init.sh' >> ~/.bashrc
WORKDIR /app
RUN git config --global --add safe.directory /app