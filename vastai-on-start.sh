#!/bin/env bash

set -e -o pipefail

mkdir /shakespeare

echo WANDB_API_KEY=$WANDB_API_KEY >> /etc/environment

echo $GOOGLE_APPLICATION_CREDENTIALS_B64 \
    | base64 --decode \
    > /shakespeare/service_account.json

echo GOOGLE_APPLICATION_CREDENTIALS_B64=$GOOGLE_APPLICATION_CREDENTIALS_B64 >> /etc/environment

echo GOOGLE_APPLICATION_CREDENTIALS=/shakespeare/service_account.json >> /etc/environment

git clone https://github.com/jeremyadamsfisher/shakespeare_transformer.git /shakespeare/shakespeare_transformer

echo PYTHONPATH=$PYTHONPATH:/shakespeare/shakespeare_transformer >> /etc/environment

echo "cd /shakespeare/shakespeare_transformer" >> ~/.bashrc