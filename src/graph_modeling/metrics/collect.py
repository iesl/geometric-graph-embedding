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
from pathlib import Path
from typing import *

import pandas as pd
import toml
from loguru import logger
from tqdm import tqdm

__all__ = [
    "collect_graph_info",
    "collect_result_info",
]


def collect_graph_info(data_path: Union[Path, str]) -> pd.DataFrame:
    """
    This function collects graph metrics from a given data path and outputs tsv files.
    It will perform a depth-first search on data_path. In each folder it will deposit a tsv file called
    "graph_metrics.tsv" where each row represents a single graph, and the columns are graph characteristics.
    For convenience, it will also save a pandas data frame as "graph_metrics.pkl".

    :param data_path: path to explore recursively
    :returns: DataFrame with aggregated statistics in data_path
    """
    data_path = Path(data_path).expanduser()

    subdir_dfs = [
        collect_graph_info(subdir) for subdir in data_path.iterdir() if subdir.is_dir()
    ]

    metrics = []
    for graph_file in data_path.glob("*[!prediction].npz"):
        graph_param = toml.load(graph_file.with_suffix(".toml"))
        graph_metrics = json.load(graph_file.with_suffix(".json").open())
        metrics.append({"path": graph_file, **graph_param, **graph_metrics})

    metrics_df = pd.DataFrame.from_records(metrics)
    aggregated_df = pd.concat((metrics_df, *subdir_dfs), ignore_index=True)
    if not aggregated_df.empty:
        aggregated_df.to_csv(data_path / "graph_metrics.tsv", sep="\t")
        aggregated_df.to_pickle(data_path / "graph_metrics.pkl")

    return aggregated_df


def collect_result_info(data_path: Union[Path, str]) -> pd.DataFrame:
    """
    This function collects results from a given data path and outputs tsv files.
    It will perform a depth-first search on data_path. In each folder it will deposit a tsv file called
    "results.tsv" where each row represents a single result, and the columns are performance and graph characteristics.
    For convenience, it will also save a pandas data frame as "results.pkl".

    :param data_path: path to explore recursively
    :returns: DataFrame with aggregated statistics in data_path
    """
    data_path = Path(data_path).expanduser()

    metrics = []
    for result_file in tqdm(data_path.glob("**/*results/**/*.metric")):
        logger.debug(result_file)
        model_config, results = open(result_file).read().split("\n")
        model_config = json.loads(model_config)
        results = json.loads(results)[0]
        seed = model_config["data_path"].split("/")[-1]
        graph_file = result_file.parent.parent.parent / f"{seed}.npz"
        try:
            graph_param = toml.load(graph_file.with_suffix(".toml"))
        except:
            graph_param = {}
        try:
            graph_metrics = json.load(graph_file.with_suffix(".json").open())
        except:
            graph_metrics = {}
        metrics.append(
            {
                "result_path": result_file,
                "path": graph_file,
                **model_config,
                **graph_param,
                **graph_metrics,
                **results,
            }
        )

    metrics_df = pd.DataFrame.from_records(metrics)
    if not metrics_df.empty:
        metrics_df.to_csv(data_path / "results.tsv", sep="\t")
        metrics_df.to_pickle(data_path / "results.pkl")

    return metrics_df
