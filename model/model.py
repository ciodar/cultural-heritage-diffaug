import copy
import importlib
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup, \
    AutoTokenizer, GenerationConfig, AutoModelForVision2Seq

from utils import rgetattr


class LitTransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            learning_rate: float = 5e-5,
            warmup_steps: int = 500,
            weight_decay: float = 0.0,
            metrics: Optional[List[dict]] = (),
            generation: Optional[dict] = None,
            freeze_text_encoder: bool = False,
            **kwargs,
    ):
        super().__init__()
        # log hyperparams in yaml
        self.save_hyperparameters()

        self.model = AutoModelForVision2Seq.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # define metrics
        self._metric_ftns = []

        # Define metrics
        for m in metrics:
            module_name, classpath = m['class_path'].split('.')[0], '.'.join(m['class_path'].split('.')[1:])
            module = importlib.import_module(module_name)
            args = m.get('init_args', {})
            metric = rgetattr(module, classpath)(**args)
            metric_name = m.get('metric_name', type(metric).__name__)
            self._metric_ftns.append((metric_name, metric))

        self.generation_cfg = GenerationConfig.from_pretrained(model_name_or_path, **generation)

        # freeze text encoder
        if freeze_text_encoder:
            for name, param in self.named_parameters():  # freeze decoder weights
                if ('git.encoder' in name) or ('git.embeddings' in name):
                    param.requires_grad = False

        # accumulators for labels and predictions
        self._labels, self._preds = [], []

    def forward(self, **inputs):
        return self.model(**inputs)

    def generate(self, pixel_values: torch.Tensor = None):
        return self.model.generate(pixel_values=pixel_values, generation_config=self.generation_cfg)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('loss/train', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # compute loss
        # See https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/36
        # TODO: do it in one pass integrated in generate.
        loss = self(**batch).loss.item()
        preds = self.generate(pixel_values=batch['pixel_values'])
        self.log('loss/validation', loss, prog_bar=True, on_step=False, on_epoch=True)
        # currently using just one reference.
        labels = batch["labels"]
        # accumulate labels and predictions to calculate metrics at the end
        self._labels.append(labels.detach().cpu())
        self._preds.append(preds.detach().cpu())
        return preds

    def on_validation_epoch_end(self) -> None:
        # shape: (batch_size, seq_len) - > (batch_size, max_seq_len)
        max_len = max([p.shape[1] for p in self._preds])
        preds = [F.pad(p, (0, max_len - p.shape[1]), "constant", self.tokenizer.pad_token_id) for p in
                 self._preds]
        preds = torch.cat(preds, dim=0)
        labels = torch.cat([y for y in self._labels]).detach().cpu().numpy()
        # decode all labels and predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for metric_name, metric in self._metric_ftns:
            self._log_metric(metric_name, metric(decoded_preds, decoded_labels))

        # reset accumulators
        self._labels, self._preds = [], []

    def configure_optimizers(self):

        parameters = [p for p in self.parameters() if p.requires_grad]
        # model = self.model
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def _log_metric(self, metric_name, metric):
        if isinstance(metric, dict):
            for k, v in metric.items():
                if isinstance(v, list):
                    v = np.mean(v)
                self.log(f"{metric_name}/{k}", v, prog_bar=True)
        else:
            self.log(metric_name, metric, prog_bar=True)
