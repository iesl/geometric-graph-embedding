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
from loguru import logger

try:
    import graph_tool as gt
    from graph_tool.generation import price_network
except ImportError as e:
    logger.warning(
        "Could not import graph_tool, did you install it? (conda install -c conda-forge graph-tool)"
    )
    raise e


def generate(
    log_num_nodes: int, seed: int, m: int, c: float, gamma: float, **kwargs
) -> nx.DiGraph:
    gt.seed_rng(seed)
    num_nodes = 2 ** log_num_nodes
    g = price_network(num_nodes, m, c, gamma)
    ngx = nx.DiGraph()
    for s, t in g.iter_edges():
        ngx.add_edge(t, s)
    return ngx
