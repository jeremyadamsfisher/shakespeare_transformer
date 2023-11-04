from bisect import bisect_left

import torch
from tqdm import tqdm


class ShiftedSequenceDataset:
    def __init__(self, config, ds):
        self.ds = ds
        self.config = config
        self.index = []
        for doc in tqdm(ds, unit="example", desc="Computing dataset index"):
            prev = self.index[-1] if self.index else -1
            n_blocks = len(doc["tokens"]) - (self.config.block_size + 1)
            self.index.append(prev + n_blocks)

    def __len__(self):
        return self.index[-1]

    def _get_idx_and_offset(self, i):
        ds_idx = bisect_left(self.index, i)
        if ds_idx == 0:
            offset = i
        else:
            offset = i - self.index[ds_idx - 1] - 1
        return ds_idx, offset

    def __getitem__(self, i):
        ds_idx, offset = self._get_idx_and_offset(i)
        tokens = self.ds[ds_idx]["tokens"]
        x = tokens[offset : offset + self.config.block_size]
        y = tokens[offset + 1 : offset + self.config.block_size + 1]
        x, y = map(torch.tensor, (x, y))
        return x, y