from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from box_training_methods.models.temps import convert_float_to_const_temp
from box_training_methods.utils import tiny_value_of_dtype
from box_training_methods import metric_logger

__all__ = [
    "InstanceAsPointEncoder",
]

class InstanceAsPointEncoder(Module):

    def __init__(self, instance_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # TODO instantiate embedding lookup layer with instances

        self.l1 = torch.nn.Linear(instance_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """

        Args:
            x: (batch_size, instance_dim)

        Returns:

        """
        h = F.relu(self.l1(x))
        instance_encoding = F.relu(self.l2(h))

        return instance_encoding
