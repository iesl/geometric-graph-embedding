from typing import *

import torch
from loguru import logger
from torch import Tensor, LongTensor, FloatTensor
from torch.nn import Module, Parameter

from pytorch_utils import TorchShape

__all__ = [
    "BoundedTemp",
    "ConstTemp",
    "GlobalTemp",
    "PerDimTemp",
    "PerEntityTemp",
    "PerEntityPerDimTemp",
    "convert_float_to_const_temp",
]


class BoundedTemp(Module):
    def __init__(
        self,
        shape: TorchShape,
        init: Union[float, Tensor] = 1.0,
        min: float = 0.0,
        max: float = 10.0,
    ):
        super().__init__()
        self.max = max
        self.min = min
        self.shape = shape
        if isinstance(init, float):
            self.init = torch.ones(shape) * init
        else:
            self.init = init
        self.temp = Parameter(
            torch.logit((self.init - self.min) / (self.max - self.min))
        )
        forward_check = self.forward()
        if not torch.allclose(
            forward_check, self.init, atol=(self.max - self.min) * 1e-8, rtol=0
        ):
            logger.warning(
                f"BoundedTemp with min={self.min}, max={self.max}, and init={self.init} has numerical issue which "
                f"results in a slightly different initialization (max error is {torch.max(self.init - forward_check)})"
            )

    def forward(
        self, idx: Union[LongTensor, slice] = slice(None, None, None)
    ) -> FloatTensor:
        return (self.max - self.min) * torch.sigmoid(self.temp[idx]) + self.min


class ConstTemp(Module):
    def __init__(self, init: float = 1.0, **kwargs):
        super().__init__()
        self.temp = Parameter(torch.tensor([init]), requires_grad=False)

    def forward(self, idxs: LongTensor) -> FloatTensor:
        """
        Return a global temp with the appropriate shape to broadcast against box tensors.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        """
        output_shape = [1] * (len(idxs.shape) + 2)
        return self.temp.view(output_shape)


class GlobalTemp(Module):
    def __init__(self, init: float, min: float, max: float, **kwargs):
        super().__init__()
        self.temp = BoundedTemp(shape=1, init=init, min=min, max=max)

    def forward(self, idxs: LongTensor) -> FloatTensor:
        """
        Return a global temp with the appropriate shape to broadcast against box tensors.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        """
        output_shape = [1] * (len(idxs.shape) + 2)
        return self.temp(0).view(output_shape)


class PerDimTemp(Module):
    def __init__(self, init: float, min: float, max: float, *, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.temp = BoundedTemp(shape=dim, init=init, min=min, max=max)

    def forward(self, idxs: LongTensor) -> FloatTensor:
        """
        Return a per-dim temp with the appropriate shape to broadcast against box tensors.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :param shape: Target shape to be capable of broadcasting with.
        """
        output_shape = [1] * (len(idxs.shape) + 2)
        output_shape[-1] = -1
        return self.temp(slice(None, None, None)).view(output_shape)


class PerEntityTemp(Module):
    def __init__(
        self, init: float, min: float, max: float, *, num_entities: int, **kwargs
    ):
        super().__init__()
        self.num_entities = num_entities
        self.temp = BoundedTemp(shape=num_entities, init=init, min=min, max=max)

    def forward(self, idxs: LongTensor) -> FloatTensor:
        """
        Return a per-dim temp with the appropriate shape to broadcast against box tensors.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        """
        output_shape = [*idxs.shape, 1, 1]
        return self.temp(idxs).view(output_shape)


class PerEntityPerDimTemp(Module):
    def __init__(
        self,
        init: float,
        min: float,
        max: float,
        *,
        num_entities: int,
        dim: int,
        **kwargs,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.temp = BoundedTemp(shape=(num_entities, dim), init=init, min=min, max=max)

    def forward(self, idxs: LongTensor) -> FloatTensor:
        """
        Return a per-entity, per-dim temp with the appropriate shape to broadcast against box tensors.
        :param idxs: Tensor of shape (..., l).
            Often, idxs may indicate edges, in which case we should have l=2, and idxs[...,0] -> idxs[..., 1] is an edge.
        """
        output_shape = [*idxs.shape, 1, -1]
        return self.temp(idxs).view(output_shape)


def convert_float_to_const_temp(temp: Union[Module, float]) -> Module:
    """Helper function to convert floats to ConstTemp modules"""
    if isinstance(temp, Module):
        return temp
    else:
        return ConstTemp(temp)
