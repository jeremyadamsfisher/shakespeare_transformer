#!/usr/bin/env bash

set -e -o pipefail

# Check if the -m flag is set
if [[ $1 == "-m" ]]; then
    INSTANCE_ID=$2
else
    echo Automatically selecting instance...
    INSTANCE=$(vastai search offers --on-demand 'gpu_name=RTX_3090 num_gpus=1 geolocation=US' --raw | jq -r 'min_by(.min_bid)')
    echo Found instance: $INSTANCE
    INSTANCE_ID=$(echo $INSTANCE | jq -r .id)
fi

vastai create instance $INSTANCE_ID \
    --image jeremyadamsfisher1123/shakespeare-gpt \
    --env "-e WANDB_API_KEY=$(cat .secrets.json | jq -r .WANDB_API_KEY) -e GOOGLE_APPLICATION_CREDENTIALS_B64=$(base64 -w 0 < ./service_account.json)" \
    --disk 100 \
    --ssh \
    --onstart ./vastai-on-start.sh