FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y direnv build-essentials
RUN eval "$(direnv hook bash)" >> /root/.bashrc
ADD gpt ./gpt
ADD setup.py makefile ./
RUN pip install -e .