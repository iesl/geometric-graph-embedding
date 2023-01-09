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

import click


@click.group()
def main():
    """Calculate and collect graph metrics / characteristics"""
    pass


@main.command(context_settings=dict(show_default=True),)
@click.argument(
    "data_path", type=click.Path(),
)
@click.option(
    "--metrics",
    "-m",
    type=click.Choice(
        [
            "all",
            "num_edges",
            "num_nodes",
            "avg_degree",
            "sparsity",
            "transitivity",
            "reciprocity",
            "flow_hierarchy",
            "clustering_coefficient",
            "assortativity",
        ],
        case_sensitive=False,
    ),
    default=("all",),
    help="name(s) of graph metric to calculate",
    multiple=True,
)
@click.option(
    "--predictions / --no_predictions",
    "-p/ ",
    default=False,
    help="calculate metrics on predictions (otherwise only calculate on original graphs)",
    multiple=False,
)
def calc(data_path, metrics, predictions=False):
    """Calculate graph metrics / characteristics"""
    from .calculate import write_metrics

    write_metrics(data_path, metrics)


@main.command(context_settings=dict(show_default=True),)
@click.argument(
    "data_path", type=click.Path(),
)
def collect_graph_info(data_path):
    """Collect graph characteristics to a single tsv"""
    from .collect import collect_graph_info

    collect_graph_info(data_path)


@main.command(context_settings=dict(show_default=True),)
@click.argument(
    "data_path", type=click.Path(),
)
def collect_result_info(data_path):
    """Collect results into a single tsv"""
    from .collect import collect_result_info

    collect_result_info(data_path)
