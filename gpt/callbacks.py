import pytorch_lightning as L
from loguru import logger

from gpt.utils import get_rank_zero_or_single_gpu


class LogGenerationPeriodically(L.Callback):
    """Log a generation from the model periodically."""

    def __init__(self, decoder, log_periodicity, wandb_logger=None):
        self.log_periodicity = log_periodicity
        self.decoder = decoder
        self.wandb_logger = wandb_logger

    def on_train_batch_start(self, trainer, model, _, batch_idx):
        if get_rank_zero_or_single_gpu():
            if batch_idx % self.log_periodicity == 0 and trainer.global_rank == 0:
                output = model.generate(max_new_tokens=50)
                output = self.decoder(output).replace("\n", " ")
                if self.wandb_logger:
                    columns = ["generation"]
                    data = [[output]]
                    self.wandb_logger.log_text(
                        "trn_generation", columns=columns, data=data
                    )
                logger.info("generation: {}", output)
