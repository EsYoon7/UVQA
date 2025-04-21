import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import is_deepspeed_available
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer import TrainerCallback
from transformers.trainer_utils import has_length

import os
import sys

def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    scores = np.mean(predictions[:,0])
    # predictions = np.argmax(predictions, axis=1)
    predictions = ((predictions[:, 0] > predictions[:, 1])==False).astype(float) # added by esyoon 2024-08-01-17:58:02
    
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy, "reward_score": scores}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)