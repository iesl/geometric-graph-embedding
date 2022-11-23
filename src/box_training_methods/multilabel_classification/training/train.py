import random
from time import time
from typing import *
import os
import toml
from pathlib import Path

import torch
from pytorch_utils import TensorDataLoader, cuda_if_available

from .dataset import MLCDataset, instances_from_pickle


def training(config: Dict) -> None:
    """
    Setup and run training loop.
    In this function we do any config manipulation required (eg. override values, set defaults, etc.)

    :param config: config dictionary
    :return: None
    """

    if config["seed"] is None:
        config["seed"] = random.randint(0, 2 ** 32)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    mlc_dataset = Path(config["data_path"])
    if mlc_dataset.is_dir():
        dataset = list({file.stem for file in mlc_dataset.glob("*.pkl")})[0]
        config["data_path"] = os.path.join(config["data_path"], str(dataset))

    if config["wandb"]:
        wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
    else:
        run_dir = Path(".")

    ### TODO currently stop at data loading but continue mimicking @mboratko's implementation
    train_dataset, train_dataloader = setup(**config)


def setup_model():
    pass


def setup_training_data(device: Union[str, torch.device], **config) -> MLCDataset:
    """
    Load the training data (pkl)

    :param device: device to load training data on
    :param config: config dictionary

    :returns: MLCDataset, ready for training
    """
    start = time()
    pkl_file = Path(config["data_path"] + ".pkl")

    if pkl_file.exists():
        train_instances, train_labels, dev_instances, dev_labels, _, _, label_encoder = instances_from_pickle(pkl_file)
        stats = toml.load(config["data_path"] + ".toml")
        num_labels = stats["num_labels"]
    else:
        raise ValueError(
            f"Could not locate training file at {config['data_path']}{{.pkl}}"
        )
    train_instances, train_labels = train_instances.to(device), train_labels.to(device)
    dev_instances, dev_labels = dev_instances.to(device), dev_labels.to(device)

    train_dataset = MLCDataset(
        train_instances, train_labels, label_encoder=label_encoder, num_labels=num_labels
    )
    dev_dataset = MLCDataset(
        dev_instances, dev_labels, label_encoder=label_encoder, num_labels=num_labels
    )

    logger.info(f"Number of labels in dataset: {dataset.num_labels:,}")
    logger.debug(f"Total time spent loading data: {time() - start:0.1f} seconds")

    return train_dataset, dev_dataset


def setup(**config):
    """
    Setup and return the datasets, dataloaders, model, and training loop required for training.

    :param config: config dictionary
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """
    device = cuda_if_available(use_cuda=config["cuda"])

    # setup data
    train_dataset, dev_dataset = setup_training_data(device, **config)
    train_dataloader = TensorDataLoader(
        train_dataset, batch_size=2 ** config["log_batch_size"], shuffle=True
    )
    return train_dataset, train_dataloader
    # TODO currently stopped at data loading but continue mimicking @mboratko's implementation
