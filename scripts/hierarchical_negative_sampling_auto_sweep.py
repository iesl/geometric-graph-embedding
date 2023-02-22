import os, random
import wandb
import yaml
from pathlib import Path


def get_graph_dirs_with_info(
        graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13"
):

    graph_dirs_with_info = []
    for d1 in os.listdir(graphs_root):
        graph_family_dir = os.path.join(graphs_root, d1)
        print(graph_family_dir)
        if os.path.isdir(graph_family_dir):
            family = d1
            for d2 in os.listdir(graph_family_dir):
                graph_dir = os.path.join(graph_family_dir, d2)
                print("\t", graph_dir)
                if os.path.isdir(graph_dir):
                    tc = d2.split("=")[-1]
                    # graph_npzs = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith(".npz")]
                    # graph_npz = random.choice(graph_npzs)
                    graph_dirs_with_info.append((family, tc, graph_dir))

    return graph_dirs_with_info


def init_sweeps():

    sweep_names = []
    sweep_ids = []

    graph_dirs_with_info = get_graph_dirs_with_info()

    for (graph_family, tc, data_path) in graph_dirs_with_info:
        for negative_sampling in ["random", "hierarchical:exact", "hierarchical:uniform", "hierarchical:descendants"]:

            config = yaml.safe_load(Path("./bin/learning/hierarchical_negative_sampling.yaml").read_text())

            config["command"].append("=".join(["--data_path", data_path]))
            if negative_sampling == "random":
                config["command"].append("=".join(["--negative_sampler", "random"]))
            else:
                negative_sampler, strategy = negative_sampling.split(":")
                config["command"].append("=".join(["--negative_sampler", "hierarchical"]))
                config["command"].append("=".join(["--hierarchical_negative_sampling_strategy", strategy]))

            sweep_name = f"{negative_sampling}_{data_path}"
            config["name"] = sweep_name
            # config["tc"] = tc
            # config["graph_family"] = graph_family
            sweep_names.append(sweep_name)

            sweep_id = wandb.sweep(sweep=config, entity="brozonoyer", project="hierarchical-negative-sampling")
            sweep_ids.append(sweep_id)

    return sweep_names, sweep_ids


if __name__ == '__main__':

    sweep_names, sweep_ids = init_sweeps()
    for i in range(len(sweep_names)):
        print(f"{sweep_names[i]}\t{sweep_ids[i]}")
