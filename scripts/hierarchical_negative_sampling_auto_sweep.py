import wandb
import yaml
from pathlib import Path


def set_command_in_config(config, command="--negative_sampler", value="random"):
    for i, c in enumerate(config["command"]):
        if c.startswith(command):
            config["command"][i] = "=".join([command, value])
    return config


def init_sweeps():

    config = yaml.safe_load(Path("./bin/learning/hierarchical_negative_sampling.yaml").read_text())

    sweep_names = []
    sweep_ids = []

    # TRANSITIVE CLOSURE
    for TC in ["True", "False"]:

        data_path = f"data/graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure={TC}/"
        config = set_command_in_config(config=config, command="--data_path", value=data_path)

        # NEGATIVE SAMPLING STRATEGIES
        for negative_sampling in ["random", "hierarchical:exact", "hierarchical:uniform", "hierarchical:descendants"]:

            if negative_sampling == "random":
                config = set_command_in_config(config=config, command="--negative_sampler", value="random")
            else:
                negative_sampler, strategy = negative_sampling.split(":")
                config = set_command_in_config(config=config, command="--negative_sampler", value=negative_sampler)
                config = set_command_in_config(config=config, command="--hierarchical_negative_sampling_strategy", value=strategy)

            sweep_name = f"TC:{TC}_{negative_sampling}"
            config["name"] = sweep_name
            sweep_names.append(sweep_name)

            sweep_id = wandb.sweep(sweep=config, entity="brozonoyer", project="hierarchical-negative-sampling")
            sweep_ids.append(sweep_id)

    return sweep_names, sweep_ids


if __name__ == '__main__':

    sweep_names, sweep_ids = init_sweeps()
    for i in range(len(sweep_names)):
        print(f"{sweep_names[i]}\t{sweep_ids[i]}")
