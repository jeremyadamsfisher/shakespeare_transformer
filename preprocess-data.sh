#!/bin/env bash

set -e -o pipefail

# This script is meant to be run on cheap cloud compute, like ec2 or google compute engine.
# You probably want to edit the data config files to point to your own bucket beforehand.

sudo apt update
sudo apt install -y python3-pip

git clone https://github.com/jeremyadamsfisher/shakespeare_transformer.git

pip install tqdm datasets PyYAML Unidecode gcsfs
pip install torch --index-url https://download.pytorch.org/whl/cpu

cd shakespeare_transformer

PYTHONPATH=. python3 gpt/convert_wikipedia.py \
--config-fp gpt/conf/data_config/wikipedia.yaml

PYTHONPATH=. python3 gpt/convert_wikipedia.py \
--config-fp gpt/conf/data_config/wikipedia_100K.yaml

PYTHONPATH=. python3 gpt/convert_wikipedia.py \
--config-fp gpt/conf/data_config/wikipedia_10K.yaml
