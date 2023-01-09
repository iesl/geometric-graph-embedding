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

import json
import time
from pathlib import Path
from typing import *

import networkx as nx
from loguru import logger
from scipy.sparse import load_npz
from tqdm import tqdm

__all__ = [
    "calc_metrics",
    "write_metrics",
]

num_nodes = lambda G: G.number_of_nodes()
num_edges = lambda G: G.number_of_edges()
sparsity = lambda G: G.number_of_edges() / G.number_of_nodes()
avg_degree = lambda G: G.number_of_edges() / G.number_of_nodes()

metric_functions = {
    "num_nodes": num_nodes,
    "num_edges": num_edges,
    "sparsity": sparsity,
    "avg_degree": avg_degree,
    "transitivity": nx.transitivity,
    "reciprocity": nx.reciprocity,
    "flow_hierarchy": nx.flow_hierarchy,
    "clustering_coefficient": nx.average_clustering,
    "assortativity": nx.degree_pearson_correlation_coefficient,
}


def calc_metrics(
    path: Union[str, Path], metrics_to_calc: Dict[str, Callable]
) -> Dict[str, Any]:
    digraph_coo = load_npz(path)
    G = nx.from_scipy_sparse_matrix(digraph_coo, create_using=nx.DiGraph)
    calculated_character = dict()
    for character, func in metrics_to_calc.items():
        time1 = time.time()
        calculated_character[character] = func(G)
        time2 = time.time()
        calculated_character[f"character_time"] = time2 - time1
    return calculated_character


def write_metrics(
    data_path: Union[str, Path],
    metrics_to_calc: Iterable[str],
    predictions: bool = False,
) -> None:
    """
    Calculate graph characteristics and write out a file
    :param data_path: path to search recursively for graphs in
    :param metrics_to_calc: names of metrics to calculate
    """
    data_path = Path(data_path).expanduser()
    unavailable_metrics = set(metrics_to_calc).difference(
        {"all", *metric_functions.keys()}
    )
    if unavailable_metrics:
        logger.warning(
            f"Requested calculation of {unavailable_metrics}, but these are not implemented."
        )
    if "all" in metrics_to_calc:
        metrics_to_calc = metric_functions
    else:
        # sort metrics_to_calc according to metric_functions
        metrics_to_calc = {
            k: v for k, v in metric_functions.items() if k in metrics_to_calc
        }
    if predictions:
        graph_files = data_path.glob("**/*prediction.npz")
    else:
        graph_files = data_path.glob("**/*[!prediction].npz")
    progress_bar = tqdm(graph_files, desc="Evaluating Metrics...")
    for graph_file in progress_bar:
        progress_bar.set_description(f"Evaluating Metrics [{graph_file}]...")
        metrics = calc_metrics(str(graph_file), metrics_to_calc)
        output_path = graph_file.with_suffix(".json")
        logger.debug(f"Writing metrics to {output_path}")
        json.dump(metrics, output_path.open("w"))
    logger.info("Complete!")
