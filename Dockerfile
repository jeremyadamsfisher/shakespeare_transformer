FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ADD gpt setup.py makefile train.sh ./
RUN pip install -e .