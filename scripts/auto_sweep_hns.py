# input: model_type, dim
# output: a list of "wandb sweep" command


import argparse
import json
import os
from pathlib import Path

import wandb


def config_generation(negative_sampling, path):
    graph_type, graph = path.split("/")[-2:]
    sweep_config = {
        "name": f"{graph_type}_{graph}_{negative_sampling}",
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


def main(config):

    if config.bayes_run == True:
        if not os.path.exists("sweeps_configs"):
            os.makedirs("sweeps_configs")
        fout = open(
            f"sweeps_configs/{config.negative_sampling}_{config.data_path.replace('/','_')}.jsonl",
            "w+",
        )
    count_sweep = 0
    count_start_sweep = 0
    count_halfway_sweep = 0
    count_ninty_sweep = 0
    count_finished_sweep = 0
    count_finished_runs = 0
    for path in Path(config.data_path).glob("**/*log_num_nodes=*"):
        count_sweep += 1
        count_this_sweep = 0
        best_hyperparams = None
        best_metric = 0.0
        target_filename = f"hns_results/{config.negative_sampling}/*metric"
        for f in path.glob(target_filename):
            # print(f)
            count_finished_runs += 1
            count_this_sweep += 1
            params, metrics = open(str(f)).read().split("\n")
            params = json.loads(params)
            metrics = json.loads(metrics)
            metric = metrics[0]["F1"]
            if metric > best_metric:
                best_hyperparams = params

        if count_this_sweep > 0:
            count_start_sweep += 1
        if count_this_sweep > config.max_run / 2:
            count_halfway_sweep += 1
        if count_this_sweep > config.max_run * 0.95:
            count_ninty_sweep += 1
        if count_this_sweep >= config.max_run:
            count_finished_sweep += 1

        print(f"{count_this_sweep} / {config.max_run} finished under {str(path)}")

        # If mode is bayes run, check each sweep directory.
        # If there is less then 95% results, clean the result and do a new sweep.
        if config.bayes_run == True and count_this_sweep <= 0.95 * config.max_run:
            print("deleting saved results in this sweep")
            target_filename = f"hns_results/{config.negative_sampling}/*metric"
            for f in path.glob(target_filename):
                f.unlink()

            sweep_config = config_generation(
                negative_sampling=config.negative_sampling, path=str(path)
            )
            sweep_id = wandb.sweep(sweep_config, entity="hierarchical-negative-sampling", project="hns")
            os.system(
                f"sh bin/launch_train_sweep.sh hierarchical-negative-sampling/hns/{sweep_id} {config.partition} {config.max_run} "
            )
            fout.write(f"{sweep_id} {json.dumps(sweep_config)}\n")

    if config.bayes_run:
        fout.close()

    print(f"# sweep started: {count_start_sweep}/{count_sweep}")
    print(f"# sweep halfway: {count_halfway_sweep}/{count_sweep}")
    print(f"# sweep 95% finished: {count_ninty_sweep}/{count_sweep}")
    print(f"# sweep finished: {count_finished_sweep}/{count_sweep}")
    print(f"# run finished: {count_finished_runs}/{count_sweep * config.max_run}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Submit multiple sweeps over slurm servers."
        + "script will only check current status if running without --best_run or --bayes_run"
    )
    parser.add_argument(
        "--bayes_run",
        action="store_true",
        help="do bayes hyper-parameter search for each sweep"
        + "(CAUTION: this will clear all existing results under each sweep data path (if not completed))",
    )
    parser.add_argument("--negative_sampling", type=str,
                        choices=["random", "hierarchical:exact", "hierarchical:uniform", "hierarchical:descendants"])
    parser.add_argument("--partition", type=str, default="gypsum-titanx")
    parser.add_argument("--max_run", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="data/graphs/")
    config = parser.parse_args()
    main(config)
