from typing import Optional, List

import torch
import torchmetrics
from lightning import LightningModule
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup, AutoConfig, \
    AutoTokenizer, AutoModelForCausalLM


class LitTransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            learning_rate: float = 5e-5,
            warmup_steps: int = 500,
            weight_decay: float = 0.0,
            train_batch_size: int = 2,
            eval_batch_size: int = 2,
            metrics: Optional[List[str]] = None,
            **kwargs,
    ):
        super().__init__()
        # log hyperparams in yaml
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # define metrics
        self._metric_ftns = [(met, getattr(torchmetrics.text, met)()) for met in metrics]
        # accumulators for labels and predictions
        self._labels, self._preds = [], []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, all_labels = batch
        preds = self.model.generate(**inputs, max_length=50)
        # TODO: include validation loss
        # currently using just one reference.
        # TODO: include all_labels
        labels = inputs["labels"]
        self._labels.append(labels.detach().cpu())
        self._preds.append(preds.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat([x for x in self._preds]).detach().cpu().numpy()
        labels = torch.cat([y for y in self._labels]).detach().cpu().numpy()
        # decode all labels and predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for name, metric in self._metric_ftns:
            self._log_metric(name, metric(decoded_preds, decoded_labels))

        # reset accumulators
        self._labels, self._preds = [], []

    def configure_optimizers(self):
        # model = self.model
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def _log_metric(self, metric_name, metric):
        if isinstance(metric, dict):
            for k, v in metric.items():
                self.log(f'val/{metric_name}_{k}', v, prog_bar=True)
        else:
            self.log('val/' + metric_name, metric, prog_bar=True)
