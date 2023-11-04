from collections import defaultdict
from random import randint


class ShiftedSequenceDataset:
    def __init__(self, config, ds):
        self.ds = ds
        self.config = config
        self.offsets = defaultdict(lambda: 0)
        self.completed = set()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """Get a random chunk"""
        tokens = self.ds[idx]["tokens"]

        if len(tokens) < self.config.block_size + 1:
            # Skip anything smaller than the context window
            raise ValueError
            return self[randint(0, len(self)) - 1]
        else:
            # TODO: keep track of the offset so we maximize coverage
            idx = randint(0, len(tokens) - self.config.block_size - 1)
            offset = self.offsets[]
            x = tokens[idx : idx + self.config.block_size]
            y = tokens[idx + 1 : idx + self.config.block_size + 1]

        return self.tokenizer.encode(x), self.tokenizer.encode(y)
