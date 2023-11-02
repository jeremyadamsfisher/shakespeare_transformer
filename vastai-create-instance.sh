#!/usr/bin/env bash

set -e -o pipefail

INSTANCE=$(vastai search offers --on-demand 'gpu_name=RTX_3090 geolocation=US' --raw | jq -r 'min_by(.min_bid)')
echo Found instance: $INSTANCE
INSTANCE_ID=$(echo $INSTANCE | jq -r .id)

vastai create instance $INSTANCE_ID \
    --image jeremyadamsfisher1123/shakespeare-gpt:0.0.6 \
    --env "-e WANDB_AUTH=$(base64 < ~/.netrc)" \
    --onstart vastai-onstart.sh \
    --disk 64 \
    --ssh