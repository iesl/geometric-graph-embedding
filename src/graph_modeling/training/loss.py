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
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from box_embeddings.common.utils import log1mexp

__all__ = [
    "BCEWithLogsLoss",
    "BCEWithLogsNegativeSamplingLoss",
    "BCEWithLogitsNegativeSamplingLoss",
    "BCEWithDistancesNegativeSamplingLoss",
    "MaxMarginWithLogitsNegativeSamplingLoss",
    "MaxMarginOENegativeSamplingLoss",
    "MaxMarginDiskEmbeddingNegativeSamplingLoss",
]


class BCEWithLogsLoss(Module):
    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        :param input: log probabilities
        :param target: target probabilities
        """
        return -(target * input + (1 - target) * log1mexp(input))


class BCEWithLogsNegativeSamplingLoss(Module):
    def __init__(self, negative_weight: float = 0.5):
        super().__init__()
        self.negative_weight = negative_weight

    def forward(self, log_prob_scores: Tensor) -> Tensor:
        """
        Returns a weighted BCE loss where:
            (1 - negative_weight) * pos_loss + negative_weight * weighted_average(neg_loss)

        :param log_prob_scores: Tensor of shape (..., 1+K) where [...,0] is the score for positive examples and [..., 1:] are negative
        :return: weighted BCE loss
        """
        log_prob_pos = log_prob_scores[..., 0]
        log_prob_neg = log_prob_scores[..., 1:]
        pos_loss = -log_prob_pos
        neg_loss = -log1mexp(log_prob_neg)
        logit_prob_neg = log_prob_neg + neg_loss
        weights = F.softmax(logit_prob_neg, dim=-1)
        weighted_average_neg_loss = (weights * neg_loss).sum(dim=-1)
        return (
            1 - self.negative_weight
        ) * pos_loss + self.negative_weight * weighted_average_neg_loss


class BCEWithLogitsNegativeSamplingLoss(Module):
    """
    Refer to NCE from Word2vec [1] + self-adversarial negative sampling [2].
    this loss optimize similarity scores.
    [1] Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality."
        Advances in neural information processing systems 26 (2013): 3111-3119.
    [2] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space."
        arXiv preprint arXiv:1902.10197 (2019).
    """

    def __init__(self, negative_weight: float = 0.5):
        super().__init__()
        self.negative_weight = negative_weight
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, logits: Tensor) -> Tensor:
        """
        Returns a weighted BCE loss where:
            (1 - negative_weight) * pos_loss + negative_weight * weighted_average(neg_loss)

        :param logit: Tensor of shape (..., 1+K) where [...,0] is the logit for positive examples
            and [..., 1:] are logits for negatives
        :return: weighted BCE loss
        """
        pos_scores = logits[..., 0]
        neg_scores = logits[..., 1:]
        pos_loss = -self.logsigmoid(pos_scores)
        neg_loss = -self.logsigmoid(-neg_scores)  # sigmoid(-x) = 1 - sigmoid(x)

        weights = F.softmax(neg_scores, dim=-1)
        weighted_average_neg_loss = (weights * neg_loss).sum(dim=-1)
        return (
            1 - self.negative_weight
        ) * pos_loss + self.negative_weight * weighted_average_neg_loss


class BCEWithDistancesNegativeSamplingLoss(Module):
    """
    Refer to RotatE [1], this loss can effectively optimize distance-based models
    [1] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational rotation in complex space."
        arXiv preprint arXiv:1902.10197 (2019).
    """

    def __init__(self, negative_weight: float = 0.5, margin=1.0):
        super().__init__()
        self.negative_weight = negative_weight
        self.margin = margin
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, distance: Tensor) -> Tensor:
        """
        :param distance: Tensor of shape (..., 1+K) where [...,0] is a distance for positive examples and [..., 1:] is
            a distance-based score for a negative example.
        :return: weighted BCE loss


        """
        pos_dists = distance[..., 0]
        neg_dists = distance[..., 1:]
        pos_loss = -self.logsigmoid(self.margin + pos_dists)
        neg_loss = -self.logsigmoid(-neg_dists - self.margin)
        weights = F.softmax(-neg_dists, dim=-1)
        weighted_average_neg_loss = (weights * neg_loss).sum(dim=-1)
        return (
            1 - self.negative_weight
        ) * pos_loss + self.negative_weight * weighted_average_neg_loss


class MaxMarginWithLogitsNegativeSamplingLoss(Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.max_margin = torch.nn.MarginRankingLoss(margin, reduction="none")

    def forward(self, logits: Tensor) -> Tensor:
        """
        Returns a max margin loss: max(0, margin - pos + neg)

        :param logits: Tensor of shape (..., 1+K) where [...,0] is the score for positive examples
            and [..., 1:] are scores for negatives
        :return: max margin loss

        """
        pos_scores = logits[..., [0]]
        neg_scores = logits[..., 1:]
        loss = self.max_margin(
            pos_scores, neg_scores, torch.ones_like(neg_scores)
        ).mean(dim=-1)
        return loss


class MaxMarginOENegativeSamplingLoss(Module):
    def __init__(self, negative_weight: float = 0.5, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, logits: Tensor) -> Tensor:
        """
        Returns a margin loss for order embedding: loss = - pos + max(0, margin + neg)

        :param logits: Tensor of shape (..., 1+K) where [...,0] is the score for positive examples
            and [..., 1:] are scores for negatives
        :return: max margin loss

        """
        pos_scores = logits[..., [0]]
        neg_scores = logits[..., 1:]
        loss = -(1 - self.negative_weight) * pos_scores.mean(
            dim=-1
        ) + self.negative_weight * torch.maximum(
            torch.zeros_like(neg_scores), self.margin + neg_scores
        ).mean(
            -1
        )
        return loss


class MaxMarginDiskEmbeddingNegativeSamplingLoss(Module):
    def __init__(self, negative_weight: float = 0.5, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, logits: Tensor) -> Tensor:
        """
        Returns a margin loss: loss = max(0, -pos) + max(0, margin + neg)
        This was recommended for HEC in https://arxiv.org/pdf/1902.04335.pdf.

        :param scores: Tensor of shape (..., 1+K) where [...,0] is the score for positive examples
            and [..., 1:] are scores for negatives. Note higher score means more positive.
        :return: max margin loss

        """
        pos_scores = logits[..., [0]]
        neg_scores = logits[..., 1:]
        loss = (1 - self.negative_weight) * (-pos_scores).clamp_min(0).mean(
            dim=-1
        ) + self.negative_weight * (self.margin + neg_scores).clamp_min(0).mean(dim=-1)
        return loss
