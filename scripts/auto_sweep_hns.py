import argparse
from pathlib import Path

import wandb


def config_generation(negative_sampling, path):
    graph_type, graph = path.split("/")[-2:]
    sweep_config = {
        "name": f"{graph_type}:::{graph}:::{negative_sampling}",
        # "controller": {"type": "local"},
        "program": "scripts/box-training-methods",
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "train",
            "${args}",
            "--task=graph_modeling",
            "--seed=12345",
            "--model_type=tbox",
            "--tbox_temperature_type=global",
            "--box_intersection_temp=0.01",
            "--box_volume_temp=1.0",
            "--log_interval=0.2",
            # "--patience=21",
            "--log_eval_batch_size=17",
            "--epochs=10000",
            "--data_path=" + path,
            "--wandb",
        ],
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "[Train] F1"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform", "min": -9.2, "max": 0},
            "dim": {"values": [8, 32, 128]},
            "negative_ratio": {"values": [8, 16, 128]},
            "log_batch_size": {"values": [0, 4, 6]},
            "negative_weight": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        },
    }

    ns = negative_sampling.split(":")
    if len(ns) == 1:
        negative_sampler = ns[0]
    else:
        negative_sampler, hierarchical_negative_sampling_strategy = ns
        sweep_config["command"].append(f"--hierarchical_negative_sampling_strategy={hierarchical_negative_sampling_strategy}")
    sweep_config["command"].append(f"--negative_sampler={negative_sampler}")

    return sweep_config


def main():

    for negative_sampling in ["random", "hierarchical:exact", "hierarchical:uniform", "hierarchical:descendants"]:
        for graph, partition in [("balanced_tree", "gypsum-titanx"), ("nested_chinese_restaurant_process", "gypsum-1080ti"), ("price", "gypsum-2080ti")]:
            for path in Path(f"data/graphs13/{graph}").glob("**/*log_num_nodes=*"):

                print(f"Initializing sweep/command for {negative_sampling}, {graph}, {partition}")

                sweep_config = config_generation(
                    negative_sampling=negative_sampling, path=str(path)
                )
                sweep_id = wandb.sweep(sweep_config, entity="hierarchical-negative-sampling", project="hns")

                with open(f"./hns_wandb_commands.sh", "a+") as f:
                    f.write(f"sh ../bin/launch_train_sweep.sh hierarchical-negative-sampling/hns/{sweep_id} {partition} 100 \n")


if __name__ == "__main__":

    main()
