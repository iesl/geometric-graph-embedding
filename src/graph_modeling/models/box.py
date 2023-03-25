# Copyright 2021 The Geometric Graph Embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from .temps import convert_float_to_const_temp
from .. import metric_logger

__all__ = [
    "BoxMinDeltaSoftplus",
    "TBox",
    "GBCBox",
    "VBCBox"
]


class BoxMinDeltaSoftplus(Module):
    def __init__(self, num_entity, dim, volume_temp=1.0, intersection_temp=1.0):
        super().__init__()
        self.centers = torch.nn.Embedding(num_entity, dim)
        self.sidelengths = torch.nn.Embedding(num_entity, dim)
        self.centers.weight.data.uniform_(-0.1, 0.1)
        self.sidelengths.weight.data.zero_()

        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060

    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.softplus(Z - z - self.softplus_const)), dim=-1,
        )
        return log_vol

    def embedding_lookup(self, idx):
        center = self.centers(idx)
        length = self.softplus(self.sidelengths(idx))
        z = center - length
        Z = center + length
        return z, Z

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1_min, e1_max = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        log_overlap_volume = self.log_volume(meet_min, meet_max)
        log_rhs_volume = self.log_volume(e2_min, e2_max)

        return log_overlap_volume - log_rhs_volume

    def forward_log_overlap_volume(self, idxs):
        """
        :param idxs: Tensor of shape (N, 2)
        :return: log of overlap volume, shape (N, )
        """
        e1_min, e1_max = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        log_overlap_volume = self.log_volume(meet_min, meet_max)

        return log_overlap_volume

    def forward_log_marginal_volume(self, idxs):
        """
        :param idxs: Tensor of shape (N, )
        :return: log of marginal volume, shape (N, )
        """
        e_min, e_max = self.embedding_lookup(idxs)
        log_volume = self.log_volume(e_min, e_max)

        return log_volume


class TBox(Module):
    """
    Box embedding model where the temperatures can (optionally) be trained.

    In this model, the self.boxes parameter is of shape (num_entity, 2, dim), where self.boxes[i,:,k] are location
    parameters for Gumbel distributions representing the corners of the ith box in the kth dimension.
        self.boxes[i,0,k] is the location parameter mu_z for a MaxGumbel distribution
        self.boxes[i,1,k] represents -mu_Z, i.e. negation of location parameter, for a MinGumbel distribution
    This rather odd convention is chosen to maximize speed / ease of computation.

    Note that with this parameterization, we allow the location parameter to "flip around", i.e. mu_z > mu_Z.
    This is completely reasonable, from the GumbelBox perspective (in fact, a bit more reasonable than requiring
    mu_Z > mu_z, as this means the distributions are no longer independent).

    :param num_entities: Number of entities to create box embeddings for (eg. number of nodes).
    :param dim: Embedding dimension (i.e. boxes will be in RR^dim).
    :param intersection_temp: Temperature for intersection LogSumExp calculations
    :param volume_temp: Temperature for volume LogSumExp calculations
        Note: Temperatures can either be either a float representing a constant (global) temperature,
        or a Module which, when called, takes a LongTensor of indices and returns their temps.
    """

    def __init__(
        self,
        num_entities: int,
        dim: int,
        intersection_temp: Union[Module, float] = 0.01,
        volume_temp: Union[Module, float] = 1.0,
    ):
        super().__init__()
        self.boxes = Parameter(
            torch.sort(torch.randn((num_entities, 2, dim)), dim=-2).values
            * torch.tensor([1, -1])[None, :, None]
        )
        self.intersection_temp = convert_float_to_const_temp(intersection_temp)
        self.volume_temp = convert_float_to_const_temp(volume_temp)

    def forward(
        self, idxs: LongTensor
    ) -> Union[Tuple[Tensor, Dict[str, Tensor]], Tensor]:
        """
        A version of the forward pass that is slightly more performant.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :returns: FloatTensor representing the energy of the edges in `idxs`
        """
        boxes = self.boxes[idxs]  # shape (..., 2, 2 (min/-max), dim)
        intersection_temp = self.intersection_temp(idxs).mean(dim=-3, keepdim=True)
        volume_temp = self.volume_temp(idxs).mean(dim=-3, keepdim=False)

        # calculate Gumbel intersection
        intersection = intersection_temp * torch.logsumexp(
            boxes / intersection_temp, dim=-3, keepdim=True
        )
        intersection = torch.max(
            torch.cat((intersection, boxes), dim=-3), dim=-3
        ).values
        # combine intersections and marginals, since we are going to perform the same operations on both
        intersection_and_marginal = torch.stack(
            (intersection, boxes[..., 1, :, :]), dim=-3
        )
        # calculating log volumes
        # keep in mind that the [...,1,:] represents negative max, thus we negate it
        log_volumes = torch.sum(
            torch.log(
                volume_temp
                * F.softplus((-intersection_and_marginal.sum(dim=-2)) / volume_temp)
                + 1e-23
            ),
            dim=-1,
        )
        out = log_volumes[..., 0] - log_volumes[..., 1]

        if self.training and isinstance(metric_logger.metric_logger, WandBLogger):
            regularizer_terms = {
                "intersection_temp": self.intersection_temp(idxs).squeeze(-2),
                "volume_temp": self.volume_temp(idxs).squeeze(-2),
                "log_marginal_vol": log_volumes[..., 1],
                "marginal_vol": log_volumes[..., 1].exp(),
                "side_length": -boxes.sum(dim=-2),
            }
            metrics_to_collect = {
                "pos": wandb.Histogram(out[..., 0].detach().exp().cpu()),
                "neg": wandb.Histogram(out[..., 1:].detach().exp().cpu()),
            }
            for k, v in regularizer_terms.items():
                metrics_to_collect[k] = wandb.Histogram(v.detach().cpu())

            metric_logger.metric_logger.collect(
                {f"[Train] {k}": v for k, v in metrics_to_collect.items()},
                overwrite=True,
            )
        return out


class GBCBox(Module):
    def __init__(self, num_entity, dim, num_universe=1.0, volume_temp=1.0, intersection_temp=1.0):
        super().__init__()
        self.num_universe = num_universe
        self.centers = torch.nn.Embedding(num_entity, dim)
        self.centers.weight.data.uniform_(-0.1, 0.1)
        self.sidelengths = torch.nn.Embedding(num_entity, dim)
        self.sidelengths.weight.data.zero_()
        self.codes = torch.nn.Embedding(self.num_universe, dim)
        torch.nn.init.uniform_(self.codes.weight, -0.1, 0.1)

        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060

    def log_volume(self, z, Z):
        log_vol_per_dim = torch.log(self.softplus(
            Z - z - self.softplus_const)).unsqueeze(-2)

        if len(log_vol_per_dim.shape) == 4:
            log_vol_per_subspace = torch.sum(
                log_vol_per_dim * self.sigmoid(self.codes.weight[None, None, :, :]), -1)  # ..., num_universe
        if len(log_vol_per_dim.shape) == 3:
            log_vol_per_subspace = torch.sum(
                log_vol_per_dim * self.sigmoid(self.codes.weight[None, :, :]), -1)  # ..., num_universe

        return log_vol_per_subspace

    def embedding_lookup(self, idx):
        center = self.centers(idx)
        length = self.softplus(self.sidelengths(idx))
        z = center - length
        Z = center + length
        return z, Z

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1_min, e1_max = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        log_overlap_volume = self.log_volume(meet_min, meet_max)
        log_rhs_volume = self.log_volume(e2_min, e2_max)
        log_conditional = log_overlap_volume - log_rhs_volume
        log_conditional = torch.max(log_conditional, -1)[0]

        return log_conditional


class VBCBox(Module):
    def __init__(self, num_entity, dim, dim_share=0, volume_temp=1.0, intersection_temp=1.0):
        super().__init__()
        self.dim_share = dim_share
        self.centers = torch.nn.Embedding(num_entity, dim)
        self.centers.weight.data.uniform_(-0.1, 0.1)
        self.sidelengths = torch.nn.Embedding(num_entity, dim)
        self.sidelengths.weight.data.zero_()
        self.codes = torch.nn.Embedding(num_entity, dim - dim_share)
        #self.ones = torch.nn.Embedding(num_entity, dim_share)
        #torch.nn.init.ones_(self.ones.weight)
        #self.ones.weight.requires_grad = False

        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060

    def log_volume(self, z, Z, c):
        # a code of near zero will make the corresponding dimension to have small affect on the final volume
        log_vol = torch.sum(
            torch.log(self.softplus(Z - z - self.softplus_const)) * c, dim=-1,
        )

        return log_vol

    def embedding_lookup(self, idx):
        center = self.centers(idx)
        length = self.softplus(self.sidelengths(idx))
        code = self.sigmoid(self.codes(idx))  # ..., dim - dim_share
        #ones = self.ones(idx)  # ..., dim_share
        ones = center[...,:self.dim_share].abs() + 1.0
        ones = ones / ones
        code = torch.cat([code, ones], axis=-1)
        z = center - length
        Z = center + length
        return z, Z, code

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1_min, e1_max, e1_code = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max, e2_code = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        code_intersection = e1_code * e2_code

        log_overlap_volume = self.log_volume(meet_min, meet_max, code_intersection)
        log_rhs_volume = self.log_volume(e2_min, e2_max, code_intersection)
        log_conditional = log_overlap_volume - log_rhs_volume

        return log_conditional
