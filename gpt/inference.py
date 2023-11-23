import tempfile

import torch
from loguru import logger

from gpt.lightning_module import GptLightning
from gpt.tokenizer import CharTokenizer


def main(checkpoint_uri, tokenizer=None, device="cpu"):
    model = GptLightning.load_from_checkpoint(checkpoint_uri)
    model.eval()
    model.to(device)

    if tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = CharTokenizer()
    while True:
        prompt = input("Prompt: ")
        prompt = tokenizer.encode(prompt)
        prompt = torch.tensor(prompt, dtype=torch.long)
        prompt = prompt.unsqueeze(0)
        prompt = prompt.to(model.device)
        output = model.generate(prompt, max_new_tokens=model.config.block_size)
        output = tokenizer.decode(output)
        print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-uri", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=False)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    args = parser.parse_args()
    main(args.checkpoint_uri, args.tokenizer, args.device)
