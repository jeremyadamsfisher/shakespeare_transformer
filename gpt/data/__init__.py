import torch
from random import randint


class ShiftedSequenceDataset:
    def __init__(self, config, ds):
        self.ds = ds
        self.config = config

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """Get a random chunk"""
        tokens = self.ds[idx]["tokens"]

        if len(tokens) < self.config.block_size + 1:
            raise ValueError("Block is too small to train the model")
        else:
            # TODO: keep track of the offset so we maximize coverage
            idx = randint(0, len(tokens) - self.config.block_size - 1)
            x = tokens[idx : idx + self.config.block_size]
            y = tokens[idx + 1 : idx + self.config.block_size + 1]

        x,y = map(torch.tensor, (x, y))
        return x, y
