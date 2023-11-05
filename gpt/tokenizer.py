import string
from typing import List, Union

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
        idxs = [self.char2idx[char] for char in s]
        return idxs

    def decode(self, idxs: Union[List[int], torch.Tensor]) -> str:
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().tolist()
        return "".join(self.idx2char[int(i)] for i in idxs)

    @property
    def vocab_size(self):
        return len(self.char2idx)
