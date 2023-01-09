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

import wandb


# load
def load_metrics(fname):
    columns = []
    path2metrics = {}

    def maybe_float(s):
        try:
            s = float(s)
        except:
            pass
        return s

    with open(fname) as fin:
        for idx, line in enumerate(fin):
            if idx == 0:
                columns = [""] + line.strip().split("\t")
                print(columns)
            else:
                splt = [maybe_float(x) for x in line.strip().split("\t")]
                path2metrics[splt[1]] = splt
    return path2metrics, columns


def load_results(fname):
    columns = []
    path2metrics = {}

    def maybe_float(s):
        try:
            s = float(s)
        except:
            pass
        return s

    with open(fname) as fin:
        for idx, line in enumerate(fin):
            if idx == 0:
                columns = [""] + line.strip().split("\t")
                print(columns)
            else:
                splt = [maybe_float(x) for x in line.strip().split("\t")]
                path2metrics[splt[2]] = splt
    return path2metrics, columns


def log_rol(k, results_dict, results_col, metrics_dict, metrics_col):
    run = wandb.init(project="icml_box_paper_v1", reinit=True)
    res = dict()
    for cname, cval in zip(results_col, results_dict[k]):
        if cname != "":
            res[cname] = cval
    for cname, cval in zip(metrics_col, metrics_dict[k]):
        if cname != "":
            res[cname] = cval
    wandb.log(res)
    run.finish()


graph_stats, graph_col = load_metrics("results/graph_metrics.tsv")

result_files = [
    "results/balanced_tree.tsv",
    "results/kronecker_graph.tsv",
    "results/nCRP.tsv",
    "results/price.tsv",
    "results/scale_free_network.tsv",
]

from tqdm import tqdm

for rf in tqdm(result_files):
    results, results_col = load_results(rf)
    for k in tqdm(results.keys()):
        log_rol(k, results, results_col, graph_stats, graph_col)
