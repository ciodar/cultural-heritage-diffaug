import functools
import importlib
from typing import Optional, List

import numpy as np
import torch
import torchmetrics
import wandb
from lightning import LightningModule
from torch.optim import AdamW

from transformers import get_cosine_schedule_with_warmup, AutoConfig, \
    AutoTokenizer, AutoModelForCausalLM

from utils import rgetattr


class LitTransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            learning_rate: float = 5e-5,
            warmup_steps: int = 500,
            weight_decay: float = 0.0,
            train_batch_size: int = 2,
            eval_batch_size: int = 2,
            metrics: Optional[List[dict]] = None,
            generation: Optional[dict] = None,
            **kwargs,
    ):
        super().__init__()
        # log hyperparams in yaml
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
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

        self.generation_cfg = {
            "max_length": int(generation.get("max_length", 100)),
            "num_beams": int(generation.get("num_beams", 1)),
            "do_sample": bool(generation.get("do_sample", False)),
            "temperature": float(generation.get("temperature", 1.0)),
            "top_k": int(generation.get("top_k", 50)),
            "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 0)),
            "early_stopping": bool(generation.get("early_stopping", False))
        }

        # accumulators for labels and predictions
        self._labels, self._preds = [], []

    def forward(self, **inputs):
        return self.model(**inputs)

    def generate(self, pixel_values: torch.Tensor = None, **generation_config):
        return self.model.generate(pixel_values=pixel_values, **generation_config)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, all_labels = batch
        preds = self.generate(pixel_values=inputs['pixel_values'], **self.generation_cfg)
        # TODO: include validation loss
        # currently using just one reference.
        # TODO: include all_labels
        labels = inputs["labels"]
        # accumulate labels and predictions to calculate metrics at the end
        self._labels.append(labels.detach().cpu())
        self._preds.append(preds.detach().cpu())
        return preds

    def on_validation_epoch_end(self) -> None:
        preds = torch.nn.utils.rnn.pad_sequence([p.transpose(0, 1) for p in self._preds],
                                                padding_value=self.tokenizer.pad_token_id)
        preds = preds.reshape(preds.shape[0], -1).transpose(0, 1)
        labels = torch.cat([y for y in self._labels]).detach().cpu().numpy()
        # decode all labels and predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for metric_name, metric in self._metric_ftns:
            self._log_metric(metric_name, metric(decoded_preds, decoded_labels))

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
                if isinstance(v, list):
                    v = np.mean(v)
                self.log(f"{metric_name}/{k}", v, prog_bar=True)
        else:
            self.log(metric_name, metric, prog_bar=True)
