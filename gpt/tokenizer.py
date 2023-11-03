import os
import string
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import wandb

BPE_TOKENIZER_FNAME = "tokenizer.json"


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        unique_chars = set(string.printable.lower() + " –")
        for idx, char in enumerate(unique_chars):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, s: str):
        idxs = torch.tensor(
            [self.char2idx[char] for char in s],
            dtype=torch.long,
        )
        return idxs

    def decode(self, idxs: list[int]) -> str:
        return "".join(self.idx2char[int(i.item())] for i in idxs)

    @property
    def vocab_size(self):
        return len(self.char2idx)


class BpeTokenzizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s).ids

    def decode(self, idxs: list[int]) -> str:
        return self.tokenizer.decode(idxs)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @classmethod
    def from_string_iterable(cls, iterable):
        with tempfile.NamedTemporaryFile("wt") as f:
            for row in iterable:
                f.write(row + "\n")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]"])
            tokenizer.train([f.name], trainer)
        return cls(tokenizer)

    @classmethod
    def from_wandb_artifact(cls, wandb_run, wandb_uri, tag=None):
        wandb_uri += tag or "latest"
        artifact = wandb_run.use_artifact(wandb_uri)
        data_dir = artifact.download()
        fp = os.path.join(data_dir, BPE_TOKENIZER_FNAME)
        tokenizer = Tokenizer.from_file(fp)
        return cls(tokenizer)

    def save_wandb_artifact(self, wandb_run):
        with tempfile.TemporaryDirectory() as tdir:
            fp = os.path.join(tdir, BPE_TOKENIZER_FNAME)
            self.tokenizer.save(fp)
            artifact = wandb.Artifact(name="shakespeare-tokenizer", type="tokenizer")
            artifact.add_file(local_path=fp)
            wandb_run.log_artifact(artifact)
