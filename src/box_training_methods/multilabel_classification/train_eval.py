import random
from time import time
from typing import *
import os
import toml
from pathlib import Path

import torch
from pytorch_utils import TensorDataLoader, cuda_if_available

from .dataset import edges_from_hierarchy_edge_list, ARFFReader, InstanceLabelsDataset
from box_training_methods.graph_modeling.dataset import RandomNegativeEdges, GraphDataset

__all__ = [
    "setup_model",
    "setup_training_data",
    "EvalLooper"
]


def setup_model():
    pass


def setup_training_data(device: Union[str, torch.device], **config) -> InstanceLabelsDataset:
    """
    Load the training data

    :param device: device to load training data on
    :param config: config dictionary

    :returns: MLCDataset ready for training
    """
    start = time()

    data_dir = Path(config["data_path"])
    assert data_dir.is_dir()
    hierarchy_edge_list_file = data_dir / "hierarchy.edgelist"

    # 1. read label taxonomy into GraphDataset
    taxnonomy_edges, label_encoder = edges_from_hierarchy_edge_list(edge_file=hierarchy_edge_list_file)
    num_labels = len(label_encoder.classes_)
    negative_sampler = RandomNegativeEdges(
        num_nodes=num_labels,
        negative_ratio=config["negative_ratio"],
        avoid_edges=None,  # TODO understand the functionality in @mboratko's code
        device=device,
        permutation_option=config["negatives_permutation_option"],
    )

    taxonomy_dataset = GraphDataset(
        taxnonomy_edges, num_nodes=num_labels, negative_sampler=negative_sampler
    )

    # 2. read instance-labels into InstanceLabelsDataset
    reader = ARFFReader(num_labels=num_labels)
    data_train = list(reader.read_internal(str(data_dir / "train-normalized.arff")))
    data_dev = list(reader.read_internal(str(data_dir / "dev-normalized.arff")))
    data_test = list(reader.read_internal(str(data_dir / "test-normalized.arff")))

    breakpoint()
    train_dataset = InstanceLabelsDataset(
        # TODO!!!
    )
    breakpoint()


    # logger.info(f"Number of edges in dataset: {dataset.num_edges:,}")
    # logger.info(f"Number of edges to avoid: {len(avoid_edges):,}")
    # logger.info(
    #     f"Number of negative edges: {num_nodes * (num_nodes - 1) - dataset.num_edges:,}"
    # )
    # logger.info(f"Density: {100*dataset.num_edges / (num_nodes * (num_nodes -1)):5f}%")
    #
    # logger.info(f"Number of labels in dataset: {dataset.num_labels:,}")
    # logger.debug(f"Total time spent loading data: {time() - start:0.1f} seconds")
    #
    # return dataset
