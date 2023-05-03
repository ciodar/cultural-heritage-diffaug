from typing import Any

import torch
import torchmetrics.functional as tmf
from torchmetrics import Metric

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def bleu_score(output,labels):
    return tmf.bleu_score(output,[labels])

def rouge_score(output,labels):
    return tmf.rouge_score(output,[labels])['rouge1_fmeasure']

def bert_score(output,labels):
    return tmf.bert_score(output,labels)['f1']
