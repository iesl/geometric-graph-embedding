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

import networkx as nx

from .generic import remove_self_loops

__all__ = [
    "generate",
]


def generate(
    log_num_nodes: int,
    alpha: float,
    gamma: float,
    delta_in: float,
    delta_out: float,
    seed: int,
    **kwargs
) -> nx.DiGraph:
    num_nodes = 2 ** log_num_nodes
    g = nx.scale_free_graph(
        num_nodes,
        alpha=alpha,
        beta=1 - alpha - gamma,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        seed=seed,
    )
    g = nx.DiGraph(g)
    g = remove_self_loops(g)
    return g
