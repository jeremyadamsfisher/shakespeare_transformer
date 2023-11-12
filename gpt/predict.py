import os

from cog import BasePredictor, Input

from gpt.train import GptLightning
from gpt.wikipedia import WikipediaDataModule


class Predictor(BasePredictor):
    def setup(self):
        checkpoint = os.environ["SHAKESPEARE_TRANSFORMER_CHECKPOINT"]
        self.lightning = GptLightning.load_from_checkpoint(self.config.checkpoint)
        dm = WikipediaDataModule(config=lightning.config)
        self.encode, self.decode = dm.encode, dm.decode

    def predict(
        self,
        text: str = Input(description="Text to continue", default="\n"),
        n_tokens: int = Input(description="Number of tokens to generate", default=1000),
    ) -> str:
        """Run a single prediction on the model"""
        self.lightning.eval()
        idxs = self.encode(text)
        idxs = self.lightning.generate(idxs)
        return self.decode(idxs)
