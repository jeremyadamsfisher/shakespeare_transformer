"""This script preprocesses wikipedia into a tokenized dataset. It should
be run before training."""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, Sequence

import yaml
from datasets import Sequence, Value, load_dataset
from torch import Tensor

from gpt.tokenizer import CharTokenizer

WIKIPEDIA_URI = "wikipedia"


def tokenize_wikipedia_dataset(
    ds,
    tokenize: Callable[[str], Tensor],
    blocksize,
):
    """Tokenize a dataset of wikipedia articles. We need to tokenize the articles
    before training because we need to know how many tokens are in each article
    to index into them."""

    def wikipedia_batch_process(batch: Dict[str, Sequence]) -> Dict[str, Sequence]:
        tokens_batch = []
        for text in batch["text"]:
            tokens = tokenize(text)
            n_blocks = len(tokens) // blocksize
            for i in range(n_blocks):
                block = tokens[i * blocksize : (i + 1) * blocksize]
                assert len(block) == blocksize
                tokens_batch.append(block)
        return {"tokens": tokens_batch}

    return ds.map(
        wikipedia_batch_process,
        batched=True,
        remove_columns=["text"],
        num_proc=mp.cpu_count() - 1,
    )


def prepare_data(n_articles, dataset_uri, tokenizer, block_size):
    if tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = CharTokenizer()

    ds = load_dataset(
        WIKIPEDIA_URI,
        "20220301.en",
        split="train",
        cache_dir=Path.cwd() / "dataset_cache",
    ).select_columns(["text"])

    if n_articles:
        ds = ds.select(range(n_articles))

    ds = tokenize_wikipedia_dataset(
        ds,
        tokenize=tokenizer.encode,
        # We need a source block that is at least one token bigger than the
        # context width of the model
        blocksize=block_size + 1,
    )

    ds = ds.cast_column("tokens", Sequence(feature=Value("int8")))

    ds = ds.train_test_split(test_size=0.0025)

    ds.save_to_disk(dataset_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-fp", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_fp) as f:
        config = yaml.safe_load(f)
    prepare_data(**config)
