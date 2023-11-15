# This is an alternate to the Cog environment to use with cloudbuild, in case
# the internet is too slow to push and pull the enormous docker image layers.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /content
ADD requirements.txt .
RUN python -m pip install -r requirements.txt && rm requirements.txt
ADD ./gpt ./gpt