import string

import torch
from unidecode import unidecode


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        unique_chars = set(string.printable.lower() + " –")
        for idx, char in enumerate(unique_chars):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, s: str):
        s = unidecode(s).lower()
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
