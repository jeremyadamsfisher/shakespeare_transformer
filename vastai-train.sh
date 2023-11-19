# !/bin/bash

set -e -o pipefail

echo "Running in the background. You may need to edit the batch size! See train.log for progress."

nohup python -O gpt/train.py \
    model_config.batch_size=32 \
    +model_config=gpt3_small_char \
    save_to=gs://shakespeare-gpt > train.log 2>&1 &