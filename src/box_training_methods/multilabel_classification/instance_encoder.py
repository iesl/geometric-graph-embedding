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

'''
class InstanceHardBoxEncoder(Module):

    def __init__(self, input_dim: int = 77, hidden_dim: int = 64, constrain_deltas_fn: str = "softplus"):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.proj_min = torch.nn.Linear(hidden_dim, 1)
        self.proj_delta = torch.nn.Linear(hidden_dim, 1)
        self.constrain_deltas_fn = constrain_deltas_fn

    def forward(self, x):
        """

        Args:
            x: (batch_size, instance_dim)

        Returns:

        """
        h = F.sigmoid(self.l1(x))
        min, delta = self.proj_min(h), self.proj_delta(h)

        if self.constrain_deltas_fn == "sqr":
            delta = torch.pow(delta, 2)
        elif self.constrain_deltas_fn == "exp":
            delta = torch.exp(delta)
        elif self.constrain_deltas_fn == "softplus":
            delta = F.softplus(delta, beta=1, threshold=20)
        elif self.constrain_deltas_fn == "proj":  # "projected gradient descent" in forward method (just clipping)
            delta = delta.clamp_min(eps)

        # TODO temperature — when temp = 0, degrades to HardBox, otherwise TBox

        max = min + delta
        instance_encoding = torch.hstack([min, max])    # (batch_size, 2)

        return instance_encoding
'''

class InstanceAsPointEncoder(Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
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
