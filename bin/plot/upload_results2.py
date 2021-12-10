import wandb
import sys


import csv
import json
import numpy as np


def load_csv(filename, metric="F1"):
    with open(filename) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.reader(csv_file, delimiter="\t", quotechar='"')
        line_count = 0
        key2id = {}
        graph2method2dim = {}
        for row in csv_reader:
            if line_count == 0:
                key2id = dict([(k, i) for i, k in list(enumerate(row))])
                # print(key2id)
                line_count += 1
                print(key2id.items())
            else:
                method_name = row[key2id["model_type"]]
                dim = int(row[key2id["dim"]])
                graph_type = row[key2id["type"]]
                transitive_closure = row[key2id["transitive_closure"]]
                # transitive_closure = row[key2id["path"]].split("transitive_closure")[1].split("/")[0][1:]
                graph_type = graph_type + "_" + str(transitive_closure)
                graph_path = row[key2id["path"]]
                # print(method_name, graph_type, dim)
                if metric == "AUC":
                    metric = float(row[key2id["AUC"]])
                else:
                    metric = float(row[key2id["F1"]])
                if graph_type not in graph2method2dim:
                    graph2method2dim[graph_type] = {}
                if method_name not in graph2method2dim[graph_type]:
                    if method_name in ["box", "vector", "complex_vector"]:
                        graph2method2dim[graph_type][method_name] = {
                            4: {},
                            16: {},
                            64: {},
                        }
                    else:
                        graph2method2dim[graph_type][method_name] = {
                            8: {},
                            32: {},
                            128: {},
                        }
                if dim not in graph2method2dim[graph_type][method_name]:
                    continue
                if graph_path not in graph2method2dim[graph_type][method_name][dim]:
                    graph2method2dim[graph_type][method_name][dim][graph_path] = -np.inf
                if metric > graph2method2dim[graph_type][method_name][dim][graph_path]:
                    graph2method2dim[graph_type][method_name][dim][graph_path] = metric
    return graph2method2dim


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


import time


def upload_results(graph2method2dim, graph_stats, graph_col):
    # graph2method2dim[graph_type][method_name][dim][graph_path]
    for graph_type in graph2method2dim.keys():
        for method_name in graph2method2dim[graph_type].keys():
            for dim in graph2method2dim[graph_type][method_name]:
                for graph_path in graph2method2dim[graph_type][method_name][dim]:
                    res = {}
                    res["graph_type"] = graph_type
                    res["method_name"] = method_name
                    if method_name == "lorentzian_distance" and (
                        dim == 8 or dim == "8"
                    ):
                        res["dim"] = dim
                        res["graph_path"] = graph_path
                        for cname, cval in zip(graph_col, graph_stats[graph_path]):
                            if cname != "":
                                res[cname] = cval
                        res["f1"] = graph2method2dim[graph_type][method_name][dim][
                            graph_path
                        ]
                        run = wandb.init(project="icml_box_paper_v11", reinit=True)
                        wandb.log(res)
                        run.finish()
                        time.sleep(2)


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


result_files = [
    "results/balanced_tree.tsv",
    "results/kronecker_graph.tsv",
    "results/nCRP.tsv",
    "results/price.tsv",
    "results/scale_free_network.tsv",
]


def f(rf):
    print(rf)
    graph_stats, graph_col = load_metrics("results/graph_metrics.tsv")
    from tqdm import tqdm

    bests = load_csv(rf)
    upload_results(bests, graph_stats, graph_col)
    return "done!"


if __name__ == "__main__":
    f(result_files[int(sys.argv[1])])
