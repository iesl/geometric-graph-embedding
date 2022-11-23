import torch
from torch.nn import Module

__all__ = [
    "OE",
    "POE",
]


class OE(Module):
    def __init__(self, num_entity, dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_entity, dim)

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1 = self.embeddings(idxs[..., 0])
        e2 = self.embeddings(idxs[..., 1])

        dist = torch.max(e1, e2) - e2
        dist = dist.square().sum(-1)

        return -dist


class POE(Module):
    def __init__(self, num_entity, dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_entity, dim)

    def log_volume(self, e):
        return -e.sum(-1)

    def intersection(self, e1, e2):
        return torch.max(e1, e2)

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1 = self.embeddings(idxs[..., 0])
        e2 = self.embeddings(idxs[..., 1])

        e_intersect = self.intersection(e1, e2)
        log_overlap_volume = self.log_volume(e_intersect)
        log_rhs_volume = self.log_volume(e2)

        return log_overlap_volume - log_rhs_volume
