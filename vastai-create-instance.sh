#!/usr/bin/env bash

set -e -o pipefail

INSTANCE=$(vastai search offers --on-demand 'gpu_name=RTX_3090 geolocation=US' --raw | jq -r 'min_by(.min_bid)')
echo Found instance: $INSTANCE
INSTANCE_ID=$(echo $INSTANCE | jq -r .id)

make docker_push

PYTHONPATH=. \
vastai create instance $INSTANCE_ID \
    --image jeremyadamsfisher1123/shakespeare-gpt:$(python -c 'import gpt; print(gpt.version)') \
    --env "-e WANDB_API_KEY=$(cat .secrets.json | jq -r .WANDB_API_KEY) -e " \
    --disk 64 \
    --ssh