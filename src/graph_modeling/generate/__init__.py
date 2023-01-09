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

from importlib import import_module
from pathlib import Path
from pprint import pformat
from typing import *

import networkx as nx
import toml
from loguru import logger
from scipy.sparse import save_npz

__all__ = [
    "write_graph",
]


def write_graph(out_dir: Union[str, Path], **graph_config):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_module = import_module("graph_modeling.generate." + graph_config["type"])

    logger.info("Generating graph with the following config:\n" + pformat(graph_config))
    tree = graph_module.generate(**graph_config)

    graph_sub_configs = []
    for name in sorted(graph_config.keys()):
        if name not in ["seed", "type"]:
            graph_sub_configs.append(f"{name}={graph_config[name]}")
    graph_folder_name = "-".join(graph_sub_configs)

    if graph_config["transitive_closure"]:
        tree = nx.transitive_closure(tree)

    logger.info("Converting to sparse matrix")
    t_scipy = nx.to_scipy_sparse_matrix(tree, format="coo")
    graph_folder = out_dir / f"{graph_config['type']}/{graph_folder_name}/"
    graph_folder.mkdir(parents=True, exist_ok=True)
    graph_file_stub = graph_folder / str(graph_config["seed"])
    logger.info(f"Saving to {graph_file_stub}")
    save_npz(graph_file_stub.with_suffix(".npz"), t_scipy)
    with graph_file_stub.with_suffix(".toml").open("w") as f:
        toml.dump(graph_config, f)
