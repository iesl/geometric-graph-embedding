import random
from time import time
from typing import *
import os
import toml
from pathlib import Path

import torch
from pytorch_utils import TensorDataLoader, cuda_if_available

from .dataset import MLCDataset

__all__ = [
    "setup_model",
    "setup_training_data",
    "EvalLooper"
]


def setup_model():
    pass


def setup_training_data(device: Union[str, torch.device], **config) -> MLCDataset:
    pass


@attr.s(auto_attribs=True)
class EvalLooper:
    name: str
    model: Module
    dl: DataLoader
    batchsize: int
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None

    @torch.no_grad()
    def loop(self) -> Dict[str, Any]:
        pass
