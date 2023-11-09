#!/usr/bin/env bash

set -e -o pipefail

if [[ ! -z "$WANDB_AUTH" ]]; then
    echo $WANDB_AUTH | base64 --decode > /root/.netrc
fi