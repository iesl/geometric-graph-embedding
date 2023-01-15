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

__all__ = [
    "generate",
]

def generate(log_num_nodes: int, **kwargs) -> nx.DiGraph:
    num_nodes = 2 ** log_num_nodes
    
    D = nx.cycle_graph(num_nodes,create_using=nx.DiGraph)
    D.add_edge(1,0)
    return D
