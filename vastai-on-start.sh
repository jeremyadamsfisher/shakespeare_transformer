#!/bin/env bash

set -e -o pipefail

## Set up dev environment

mkdir /shakespeare
echo $GOOGLE_APPLICATION_CREDENTIALS_B64 \
    | base64 --decode \
    > /shakespeare/service_account.json
export GOOGLE_APPLICATION_CREDENTIALS=/shakespeare/service_account.json
git clone https://github.com/jeremyadamsfisher/shakespeare_transformer.git /shakespeare/shakespeare_transformer
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
cd /shakespeare/shakespeare_transformer
gsutil -m cp -r gs://shakespeare-gpt/char_tokenized_wikipedia_gpt3 .
mv char_tokenized_wikipedia_gpt3 wikipedia_ds

## Add creds for later, if desired

echo WANDB_API_KEY=$WANDB_API_KEY >> /etc/environment
echo GOOGLE_APPLICATION_CREDENTIALS_B64=$GOOGLE_APPLICATION_CREDENTIALS_B64 >> /etc/environment
echo GOOGLE_APPLICATION_CREDENTIALS=/shakespeare/service_account.json >> /etc/environment
echo PYTHONPATH=$PYTHONPATH:/shakespeare/shakespeare_transformer >> /etc/environment
echo "cd /shakespeare/shakespeare_transformer" >> ~/.bashrc
echo "gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS" >> ~/.bashrc

## Run the training!

export PYTHONPATH=$PYTHONPATH:/shakespeare/shakespeare_transformer
nohup python -O gpt/train.py \
    model_config.batch_size=32 \
    +model_config=gpt3_small_char \
    +data_config=wikipedia \
    save_to=gs://shakespeare-gpt &> train.log &