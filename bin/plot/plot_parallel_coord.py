def plotme_single_f1(np_mat, ynames, title):
    """
  np_mat - N rows by C columns, all values to plot (real valued)
  ynames - C items, strings naming the C columns of np_mats
  title - 1 title for the plot
  """
    # https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches
    import numpy as np

    ys = np_mat

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(10, 4))
    from matplotlib import cm

    viridis = cm.get_cmap("viridis", 256)

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        host.set_xticklabels(ynames, fontsize=14)
        host.tick_params(axis="x", which="major", pad=7)
        host.spines["right"].set_visible(False)
        host.xaxis.tick_top()
        host.set_title(title, fontsize=18)

        colors = plt.cm.Set2.colors
        # each row the matrix
        for j in range(ys.shape[0]):
            # to just draw straight lines between the axes:
            # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(
                zip(
                    [
                        x
                        for x in np.linspace(
                            0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True
                        )
                    ],
                    np.repeat(zs[j, :], 3)[1:-1],
                )
            )
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path, facecolor="none", lw=1, edgecolor=viridis(ys[j, ys.shape[1] - 1])
            )
            host.add_patch(patch)
        # host.legend(legend_handles, legend_names,
        #       loc='lower center', bbox_to_anchor=(0.5, -0.18),
        #       ncol=len(model_names), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig("/tmp/parallel_coordinates.pdf")


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

    numpy_mat_cols = [
        "sparsity",
        "avg_degree",
        "transitivity",
        "assortativity",
        "reciprocity",
        "flow_hierarchy",
        "branching",
    ]
    numpy_mat_cols = ["sparsity", "avg_degree", "transitivity", "reciprocity"]
    import collections

    method2dim2rows = collections.defaultdict(dict)
    numpy_mat = dict()  # method2dim2res
    for graph_type in graph2method2dim.keys():
        for method_name in graph2method2dim[graph_type].keys():
            for dim in graph2method2dim[graph_type][method_name]:
                for graph_path in graph2method2dim[graph_type][method_name][dim]:
                    if dim not in method2dim2rows[method_name]:
                        method2dim2rows[method_name][dim] = []
                    try:
                        res = []
                        stats = dict(
                            [
                                (cname, cval)
                                for cname, cval in zip(
                                    graph_col, graph_stats[graph_path]
                                )
                            ]
                        )
                        for c in numpy_mat_cols:
                            print(stats[c], c)
                            res.append(float(stats[c]))
                        res.append(
                            graph2method2dim[graph_type][method_name][dim][graph_path]
                        )  # f1
                        method2dim2rows[method_name][dim].append(res)
                    except:
                        print("error processing one line %s" % str(graph_path))
    return method2dim2rows, numpy_mat_cols


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
    method2dim2rows, columns = upload_results(bests, graph_stats, graph_col)
    for method_name in method2dim2rows:
        print(method_name)
        for dim in method2dim2rows[method_name]:
            print(dim)
            results = np.array(method2dim2rows[method_name][dim])
            plotme_single_f1(results, columns + ["f1"], rf)
    return "done!"


if __name__ == "__main__":
    # f(result_files[int(sys.argv[1])])
    f("results/balanced_tree.tsv")
