from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from .temps import convert_float_to_const_temp
from utils import tiny_value_of_dtype
import metric_logger

__all__ = [
    "BoxMinDeltaSoftplus",
    "TBox",
    "HardBox",
]


eps = tiny_value_of_dtype(torch.float)


# TODO rename to BoxCenterDeltaSoftplus
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


class HardBox(Module):
    """
    https://arxiv.org/pdf/1805.04690.pdf

    To be used with non-measure-theoretic loss functions such as push-pull loss
    """

    def __init__(
        self,
        num_entities: int,
        dim: int,
        constrain_deltas_fn: str
    ):
        super().__init__()

        self.U = Parameter(torch.randn((num_entities, dim)))  # parameter for min
        self.V = Parameter(torch.randn((num_entities, dim)))  # unconstrained parameter for delta

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

        if self.training:

            # produce box embeddings to be used in push-pull loss
            return torch.stack([mins, deltas], dim=-2)

        else:  # self.eval

            yu, yv, xu, xv = mins[..., [0], :], deltas[..., [0], :], mins[..., [1], :], deltas[..., [1], :]  # (bsz, 1, dim)
            yz, yZ, xz, xZ = yu, yu + yv, xu, xu + xv  # (bsz, 1, dim)

            # compute hard intersection
            z = torch.max(torch.cat([yz, xz], dim=-2), dim=-2)[0]  # (bsz, 1, dim)
            Z = torch.min(torch.cat([yZ, xZ], dim=-2), dim=-2)[0]  # (bsz, 1, dim)

            # log(Π(d)) -> Σ(log(d))
            # do clamp_min over relu (non-negative box dimensions) so that log doesn't go to -inf
            y_intersection_x_log_vol = torch.squeeze(
                torch.sum(
                    torch.log(
                        F.relu(Z-z).clamp_min(eps)
                    ),
                    dim=-1, keepdim=True
                )
            )
            """
            one-line for debugging:
            torch.squeeze(torch.sum(torch.log(F.relu(Z-z).clamp_min(eps)), dim=-1, keepdim=True))
            """

            x_log_vol = torch.squeeze(
                torch.sum(
                    torch.log(
                        F.relu(xZ - xz).clamp_min(eps)
                    ),
                    dim=-1, keepdim=True
                )
            )

            # energy := -log(P(y|x)) = -log(V(y&x)/V(x)) = logV(x) - logV(y&x)
            energy = x_log_vol - y_intersection_x_log_vol  # should be 0 if y contains x
            threshold = 0.1
            containment = torch.le(energy, threshold).int()  # if entry is 0, y > x

            return containment
