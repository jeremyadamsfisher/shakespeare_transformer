from argparse import Namespace

from gpt.wikipedia import ShiftedSequenceDataset


def test_shifted_seq_index():
    # Create 3 K-length documents

    K = 6
    block_size = 2
    n_examples = 3
    examples = [{"tokens": [0] * K} for _ in range(n_examples)]
    config = Namespace(block_size=block_size)
    n_usable_blocks_per_example = K - block_size

    assert n_usable_blocks_per_example == 4

    """E.g.:
    |foobar|
    |__    |
    | __   |
    |  __  |
    |   __ |
    |    __| <- this one is unusable b/c there is no label, so there are 4 usables
    """

    ds = ShiftedSequenceDataset(config, examples)

    assert ds.index == [
        n_usable_blocks_per_example * i for i in range(1, n_examples + 1)
    ]

    expected = [
        (i // n_usable_blocks_per_example, i % n_usable_blocks_per_example)
        for i in range(n_examples * n_usable_blocks_per_example)
    ]
    actual = [
        ds._get_idx_and_offset(i)
        for i in range(n_examples * n_usable_blocks_per_example)
    ]

    assert actual == expected
