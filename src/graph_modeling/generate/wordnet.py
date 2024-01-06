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

from itertools import chain
import networkx as nx
from nltk.corpus import wordnet

__all__ = [
    "generate",
]


def generate(log_num_nodes: int, seed: int, root_name: str, **kwargs) -> nx.DiGraph:
    G = nx.DiGraph()
    queue = wordnet.synsets(root_name)
    if not queue:
        raise ValueError(f"Synset with name '{root_name}' does not exist")
    if len(queue) > 1:
        raise ValueError(f"More than one synset matches name '{root_name}'.")

    node_limit = 2 ** log_num_nodes
    G.add_node(queue[0])
    while queue:
        node = queue.pop(0)
        for hyponym in chain(node.hyponyms(), node.instance_hyponyms()):
            if G.number_of_nodes() >= node_limit:
                queue = []
                break
            G.add_edge(node, hyponym)
            queue.append(hyponym)

    return G
