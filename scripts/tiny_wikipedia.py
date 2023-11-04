from datasets import Dataset, load_dataset
from tqdm import tqdm

print("Loading the English Wikipedia dataset...")
wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")

print("Creating a smaller subset of 10,000 rows...")
texts = []
for row in tqdm(wiki_dataset, total=10_000):
    text = row["text"]
    texts.append(text)
    if len(texts) == 10_000:
        break

smaller_dataset = Dataset.from_dict({"text": texts})

print("Uploading to the Huggingface Hub...")
smaller_dataset.push_to_hub("jeremyf/tiny_wikipedia_en")
