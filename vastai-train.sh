# !/bin/bash

set -e -o pipefail

echo "You may need to edit the batch size!"

nohup python -O gpt/train.py \
    ++model_config.batch_size=1024 \
    save_to=gs://shakespeare-gpt \
    +model_config=baby \
    n_articles=100 > train.log 2>&1 &
