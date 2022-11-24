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
