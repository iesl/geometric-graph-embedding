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
from torch.nn import Module, Parameter

__all__ = [
    "VectorSim",
    "VectorDist",
    "BilinearVector",
    "ComplexVector",
]


class VectorSim(Module):
    def __init__(self, num_entity, dim, separate_io=True, use_bias=False):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        if separate_io:
            self.embeddings_out = torch.nn.Embedding(num_entity, dim)
        else:
            self.embeddings_out = self.embeddings_in
        if use_bias == True:
            self.bias = torch.nn.Parameter(torch.zeros(1,))
        else:
            self.bias = 0.0

    def forward(self, idxs):
        e1 = self.embeddings_in(idxs[..., 0])
        e2 = self.embeddings_out(idxs[..., 1])
        logits = torch.sum(e1 * e2, dim=-1) + self.bias
        return logits


class VectorDist(Module):
    def __init__(self, num_entity, dim, separate_io=True):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        if separate_io:
            self.embeddings_out = torch.nn.Embedding(num_entity, dim)
        else:
            self.embeddings_out = self.embeddings_in

    def forward(self, idxs):
        e1 = self.embeddings_in(idxs[..., 0])
        e2 = self.embeddings_out(idxs[..., 1])
        log_probs = -torch.sum(torch.square(e1 - e2), dim=-1)
        return log_probs


class BilinearVector(Module):
    def __init__(self, num_entity, dim, separate_io=True, use_bias=False):
        super().__init__()
        self.embeddings_in = torch.nn.Embedding(num_entity, dim)
        if separate_io:
            self.embeddings_out = torch.nn.Embedding(num_entity, dim)
        else:
            self.embeddings_out = self.embeddings_in
        self.bilinear_layer = torch.nn.Bilinear(dim, dim, 1, use_bias)
        self.use_bias = use_bias

    def forward(self, idxs):
        e1 = self.embeddings_in(idxs[..., 0])
        e2 = self.embeddings_out(idxs[..., 1])
        logits = self.bilinear_layer(e1, e2).squeeze(-1)
        return logits


class ComplexVector(Module):
    def __init__(self, num_entity, dim):
        super().__init__()
        self.embeddings_re = torch.nn.Embedding(num_entity, dim)
        self.embeddings_im = torch.nn.Embedding(num_entity, dim)
        self.w = Parameter(torch.randn((2, dim)))

    def forward(self, idxs):
        entities_re = self.embeddings_re(idxs)  # (..., 2, dim)
        entities_im = self.embeddings_im(idxs)  # (..., 2, dim)
        s_r, s_i = entities_re[..., 0, :], entities_im[..., 0, :]
        o_r, o_i = entities_re[..., 1, :], entities_im[..., 1, :]

        logits = (
            (s_r * self.w[0] * o_r).sum(-1)
            + (s_i * self.w[0] * o_i).sum(-1)
            + (s_r * self.w[1] * o_i).sum(-1)
            - (s_i * self.w[1] * o_r).sum(-1)
        )
        return logits
