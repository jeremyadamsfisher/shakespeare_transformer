# This is an alternate to the Cog environment to use with cloudbuild, in case
# the internet is too slow to push and pull the enormous docker image layers.
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
WORKDIR /content
ADD requirements.txt .
RUN python -m pip install -r requirements.txt && rm requirements.txt
ADD ./gpt ./gpt