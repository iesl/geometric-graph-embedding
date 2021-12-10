# input: model_type, dim
# output: a list of "wandb sweep" command


import argparse
import json
import os
from pathlib import Path

import wandb


def config_generation(model_type, dim, path):

    sweep_config = {
        # "controller": {"type": "local"},
        "program": "scripts/graph-modeling",
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "train",
            "${args}",
            "--model_type=" + model_type,
            "--log_interval=0.1",
            "--patience=21",
            "--log_eval_batch_size=17",
            "--epochs=10000",
            "--negative_ratio=128",
            "--dim=" + str(dim),
            "--data_path=" + path,
        ],
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "[Train] F1"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform", "min": -9.2, "max": 0},
            "log_batch_size": {"distribution": "int_uniform", "min": 8, "max": 11},
            "negative_weight": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        },
    }

    if model_type in [
        "box",
        "global_learned_temp_box",
        "per_entity_learned_temp_box",
        "per_dim_learned_temp_box",
        "pure_gumbel",
    ]:
        sweep_config["parameters"]["box_intersection_temp"] = {
            "distribution": "log_uniform",
            "min": -9.2,
            "max": -0.69,
        }
        sweep_config["parameters"]["box_volume_temp"] = {
            "distribution": "log_uniform",
            "min": -2.3,
            "max": 2.3,
        }
    if model_type == "oe":
        sweep_config["parameters"]["margin"] = {
            "distribution": "uniform",
            "min": 0,
            "max": 10,
        }
    if (
        model_type == "lorentzian_distance"
        or model_type == "lorentzian"
        or model_type == "lorentzian_score"
    ):
        sweep_config["parameters"]["lorentzian_alpha"] = {
            "distribution": "uniform",
            "min": 0.0,
            "max": 10.0,
        }
        sweep_config["parameters"]["lorentzian_beta"] = {
            "distribution": "uniform",
            "min": 0.0,
            "max": 10,
        }
    if model_type == "vector_dist":
        sweep_config["parameters"]["margin"] = {
            "distribution": "uniform",
            "min": 1,
            "max": 30,
        }
        sweep_config["command"].append("--separate_io")
    if model_type == "vector":
        sweep_config["command"].append("--separate_io")
    if model_type == "bilinear_vector":
        sweep_config["command"].append("--no_separate_io")

    return sweep_config


def main(config):

    if config.model_type not in [
        "box",
        "per_entity_learned_temp_box",
        "per_dim_learned_temp_box",
        "global_learned_temp_box",
        "pure_gumbel",
        "vector",
        "complex_vector",
        "vector_dist",
    ]:
        config.dim = config.dim * 2

    if config.bayes_run == True:
        if not os.path.exists("sweeps_configs"):
            os.makedirs("sweeps_configs")
        fout = open(
            f"sweeps_configs/{config.model_type}_{config.dim}_{config.data_path.replace('/','_')}.jsonl",
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
        target_filename = f"results/{config.model_type}_{config.dim}/*metric"
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
            target_filename = f"results/{config.model_type}_{config.dim}/*metric"
            for f in path.glob(target_filename):
                f.unlink()

            sweep_config = config_generation(
                model_type=config.model_type, dim=config.dim, path=str(path)
            )
            sweep_id = wandb.sweep(sweep_config, project="learning_generated_graph")
            os.system(
                f"sh bin/launch_train_sweep.sh dongxu/learning_generated_graph/{sweep_id} {config.partition} {config.max_run} "
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
    parser.add_argument("--model_type", type=str)
    parser.add_argument(
        "--dim",
        type=int,
        help="dimension will double when the model_type is not box or vector",
    )
    parser.add_argument("--partition", type=str, default="titanx-short")
    parser.add_argument("--max_run", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="data/graphs/")
    config = parser.parse_args()
    if config.model_type not in [
        "box",
        "global_learned_temp_box",
        "per_entity_learned_temp_box",
        "per_dim_learned_temp_box",
        "pure_gumbel",
        "oe",
        "poe",
        "vector",
        "vector_dist",
        "bilinear_vector",
        "complex_vector",
        "lorentzian_distance",
        "lorentzian_score",
        "lorentzian",
    ]:
        raise Exception(f"model type {config.model_type} does not exist")
    main(config)
