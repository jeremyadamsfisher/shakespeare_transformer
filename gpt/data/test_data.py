from argparse import Namespace

from gpt.data import ShiftedSequenceDataset


def test_shifted_seq_index():
    K = 6
    block_size = 2
    seq = [{"tokens": [K * i + j for j in range(K)]} for i in range(100)]
    config = Namespace(block_size=block_size)
    n_blocks_per_example = K - block_size - 1
     
    print("n_blocks_per_example: ", n_blocks_per_example)

    ds = ShiftedSequenceDataset(config, seq)

    m = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for i, (idx, offset) in enumerate(m):
        assert ds._get_idx_and_offset(i) == (idx, offset)