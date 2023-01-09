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
