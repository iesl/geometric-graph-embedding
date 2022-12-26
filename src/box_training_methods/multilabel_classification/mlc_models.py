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
    "HardBox",
]


class InstanceEncoder(Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class HardBox(Module):

    def __init__(
        self,
        num_labels: int,
        dim: int,
        constrain_deltas_fn: str
    ):
        super().__init__()

        self.U = Parameter(torch.randn((num_labels, dim)))  # parameter for min
        self.V = Parameter(torch.randn((num_labels, dim)))  # unconstrained parameter for delta

        self.constrain_deltas_fn = constrain_deltas_fn  # sqr, exp, softplus, proj

    def forward(
        self, idxs: LongTensor
    ) -> Union[Tuple[Tensor, Dict[str, Tensor]], Tensor]:
        """
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        """

        # (bsz, K+1 (+/-), 2 (y > x), dim) if train
        # (bsz, 2 (y > x), dim) if inference
        mins = self.U[idxs]
        deltas = self.V[idxs]  # deltas must be > 0
        if self.constrain_deltas_fn == "sqr":
            deltas = torch.pow(deltas, 2)
        elif self.constrain_deltas_fn == "exp":
            deltas = torch.exp(deltas)
        elif self.constrain_deltas_fn == "softplus":
            deltas = F.softplus(deltas, beta=1, threshold=20)
        elif self.constrain_deltas_fn == "proj":  # "projected gradient descent" in forward method (just clipping)
            deltas = deltas.clamp_min(eps)

        # produce box embeddings to be used in push-pull loss
        return torch.stack([mins, deltas], dim=-2)
