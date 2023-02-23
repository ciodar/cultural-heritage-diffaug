import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor


def get_pretrained_model(name_or_path):
    model, processor = AutoModelForCausalLM.from_pretrained(name_or_path), AutoProcessor.from_pretrained(name_or_path)
    return model, processor
