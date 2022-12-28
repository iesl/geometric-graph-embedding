from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from .temps import convert_float_to_const_temp
from box_training_methods.utils import tiny_value_of_dtype
from box_training_methods import metric_logger

__all__ = [
    "InstanceEncoder",
]


class InstanceEncoder(Module):

    def __init__(self, input_dim=77, hidden_dim=64):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.proj_min = torch.nn.Linear(hidden_dim, 1)
        self.proj_delta = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        """

        Args:
            X: (batch_size, instance_dim)

        Returns:

        """
        h = F.sigmoid(self.l1(X))
        min, delta = self.proj_min(h), self.proj_delta(h)
        breakpoint()
        return min, delta


class Scorer(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, label_one_hots, label_boxes):
        """

        Args:
            x: vector or box embedding of instance (produced by InstanceEncoder):       (..., 2 [if box])
            label_one_hots: one-hot vectors representing true labels for each instance: (..., num_labels)
            label_boxes: boxes for each label in taxonomy:                              (..., 2 [min/max])

        Returns: nll

        """
