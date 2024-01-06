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

import functools
import random

import click

from . import write_graph


@click.group()
def main():
    """Graph generation commands"""
    pass


def _common_options(func):
    """Common options used in all subcommands"""

    @main.command(context_settings=dict(show_default=True))
    @click.option(
        "--outdir",
        default="data/graphs/",
        type=click.Path(writable=True),
        help="location to save output",
    )
    @click.option(
        "--log_num_nodes", type=int, default=12, help="2**log_num_nodes number of nodes"
    )
    @click.option("--seed", type=int, default=None, help="manually set random seed")
    @click.option(
        "--transitive_closure / --no_transitive_closure",
        default=False,
        help="create the transitive closure of the generated graph",
    )
    @functools.wraps(func)
    def wrapper(*args, seed, **kwargs):
        if seed is None:
            seed = random.randint(0, 2 ** 32)
        return func(*args, seed=seed, **kwargs)

    return wrapper


@_common_options
@click.option("--branching", default=2, help="branching factor")
def balanced_tree(outdir, **graph_config):
    """Writes out a balanced directed tree"""
    write_graph(outdir, type="balanced_tree", **graph_config)


@_common_options
def random_tree(outdir, **graph_config):
    """Writes out a random directed tree"""
    write_graph(outdir, type="random_tree", **graph_config)


@_common_options
@click.option(
    "--alpha",
    default=0.41,
    help="probability for adding a new node connected to an existing node chosen randomly according "
    "to the in-degree distribution (0 <= alpha + gamma <= 1)",
)
@click.option(
    "--gamma",
    default=0.05,
    help="probability for adding a new node connected to an existing node chosen randomly according "
    "to the out-degree distribution (0 <= alpha + gamma <= 1)",
)
@click.option(
    "--delta_in",
    default=0.2,
    help="bias for choosing nodes from in-degree distribution",
)
@click.option(
    "--delta_out",
    default=0.0,
    help="bias for choosing nodes from out-degree distribution",
)
def scale_free_network(outdir, **graph_config):
    """Writes out a scale-free directed graph"""
    write_graph(outdir, type="scale_free_network", **graph_config)


@_common_options
@click.option(
    "--alpha",
    default=10,
    help="probability of adding a new table is proportional to alpha (>0)",
)
def ncrp(outdir, **graph_config):
    """Writes out a nested Chinese restaurant process graph"""
    write_graph(outdir, type="nested_chinese_restaurant_process", **graph_config)


@_common_options
@click.option(
    "--a", default=1.0, help="first entry of seed graph",
)
@click.option(
    "--b", default=0.6, help="second entry of seed graph",
)
@click.option(
    "--c", default=0.5, help="third entry of seed graph",
)
@click.option(
    "--d", default=0.2, help="fourth entry of seed graph",
)
def kronecker(outdir, **graph_config):
    """Writes out a Kronecker graph"""
    write_graph(outdir, type="kronecker_graph", **graph_config)


@_common_options
@click.option(
    "--m", default=1, help="Out-degree of newly added vertices.",
)
@click.option(
    "--c",
    default=1.0,
    help="Constant factor added to the probability of a vertex receiving an edge",
)
@click.option(
    "--gamma", default=1.0, help="Preferential attachment exponent",
)
def price(outdir, **graph_config):
    """Writes out a graph produced using the Price model"""
    write_graph(outdir, type="price", **graph_config)


@_common_options
@click.option(
    "--vector_file", default="", help="fourth entry of seed graph",
)
def hac(outdir, **graph_config):
    """Writes out a HAC graph"""
    write_graph(outdir, type="hac", **graph_config)


@_common_options
@click.option(
    "--vector_file", default="", help="xcluster format",
)
@click.option(
    "--k", default=5, help="number of neighbors",
)
def knn_graph(outdir, **graph_config):
    """Writes out a KNN graph"""
    write_graph(outdir, type="knn_graph", **graph_config)


@_common_options
@click.option(
    "--root_name", default="entity", help="Name of root node to start traversing from",
)
@click.option(
    "--traversal_method", default="dfs", help="How to expand from the root [dfs or bfs]"
)
def wordnet(outdir, **graph_config):
    write_graph(outdir, type="wordnet", **graph_config)
