#!/bin/env bash

mkdir /shakespeare

echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc"

echo $GOOGLE_APPLICATION_CREDENTIALS_B64 \
    | base64 --decode \
    > /shakespeare/service_account.json

echo "export GOOGLE_APPLICATION_CREDENTIALS=/shakespeare/service_account.json" >> ~/.bashrc

git clone https://github.com/jeremyadamsfisher/shakespeare_transformer.git /shakespeare/shakespeare_transformer

echo "cd /shakespeare/shakespeare_transformer" >> ~/.bashrc