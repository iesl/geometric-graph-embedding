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

import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
from loguru import logger

from .generic import remove_self_loops

__all__ = [
    "generate",
]


FILE_DIR = Path(os.path.realpath(__file__)).parent
KRONECKER_BINARY_LOCATION = FILE_DIR / "../../../libc/snap/examples/krongen/krongen"

if not KRONECKER_BINARY_LOCATION.exists():
    logger.warning(
        f"Cannot locate Kronecker generation binary, did you compile it? (cd libc/snap/examples/krongen/; make all)"
    )
    raise RuntimeError(
        f"Kronecker generation binary not found at {KRONECKER_BINARY_LOCATION.resolve()}"
    )


def generate(
    log_num_nodes: int, a: float, b: float, c: float, d: float, seed: int, **kwargs
) -> nx.DiGraph:

    tmp_graph_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_graph_file.close()
    logger.info(f"Generating graph in temporary file {tmp_graph_file.name}")
    args = [
        str(KRONECKER_BINARY_LOCATION),
        f"-o:{tmp_graph_file.name}",
        f'-m:"{a} {b}; {c} {d}"',
        f"-i:{log_num_nodes}",
        "-s:{seed}",
    ]
    logger.info(f"Running subprocess {' '.join(args)}")
    subprocess.call(args)
    logger.info(f"Subprocess krongen completed, reading edge list")
    edge_list = []
    with open(tmp_graph_file.name, "r") as f:
        for line in f.read().split("\n")[4:-1]:
            e1, e2 = line.split("\t")
            edge_list.append((int(e1), int(e2)))
    g = nx.DiGraph(edge_list)
    logger.info(
        f"Generated graph has {g.number_of_nodes()} nodes, {g.number_of_edges()} edges"
    )
    logger.info(f"Removing self-loops")
    g = remove_self_loops(g)
    logger.info(f"After removing self-loops, graph has {g.number_of_edges()} edges")
    os.unlink(tmp_graph_file.name)
    return g
