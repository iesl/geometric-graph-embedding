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
