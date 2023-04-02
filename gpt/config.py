from pydantic import BaseModel


class GptConfig(BaseModel):
    p_dropout = 0.2
    batch_size = 1024
    block_size = 32
    n_embed = 32
    vocab_size = 65
    n_heads = 4
    n_gpt_blocks = 3
    test_train_split = 0.1
    lr = 1e-3
    n_layers = 3
    n_epochs = 1


gpt_small = GptConfig()
