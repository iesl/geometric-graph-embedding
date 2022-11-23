import math
from pathlib import Path
from time import time
import pickle
from typing import *

import attr
import numpy as np
import torch
from loguru import logger
from torch import Tensor, LongTensor
from torch.utils.data import Dataset

from sklearn import preprocessing


__all__ = [
    "MLCDataset",
]


def instances_from_pickle(path: Union[str, Path]):
    """
    Loads instances and number of labels from a pkl file. Meant for importing multilabel classification data.

    :param path: Location of pkl file.
    :returns: Pytorch LongTensor of x,y and int representing number of labels in dataset
    """
    start = time()
    logger.info(f"Loading {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    train_instances, train_labels = [p["x"] for p in data["train"]], [p["labels"] for p in data["train"]]
    dev_instances, dev_labels = [p["x"] for p in data["dev"]], [p["labels"] for p in data["dev"]]
    test_instances, test_labels = [p["x"] for p in data["test"]], [p["labels"] for p in data["test"]]
    logger.info(f"Creating PyTorch LongTensor representation of instances...")
    train_instances, dev_instances, test_instances = torch.tensor(train_instances), torch.tensor(dev_instances), torch.tensor(test_instances)
    # TODO figure out how to properly encode labels -- preserve hierarchy or not?
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit([l for ls in train_labels + dev_labels + test_labels for l in ls])
    train_labels = [label_encoder.transform(ls) for ls in train_labels]
    dev_labels = [label_encoder.transform(ls) for ls in dev_labels]
    test_labels = [label_encoder.transform(ls) for ls in test_labels]
    breakpoint()
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return train_instances, train_labels, dev_instances, dev_labels, test_instances, test_labels, label_encoder


@attr.s(auto_attribs=True)
class MLCDataset(Dataset):
    """
    """

    instances: Tensor# = attr.ib(validator=_validate_edge_tensor)
    labels: Tensor
    num_labels: int
    label_encoder: preprocessing.LabelEncoder

    def __attrs_post_init__(self):
        self._device = self.instances.device

    def __getitem__(self, idxs: LongTensor) -> LongTensor:
        """
        :param idxs: LongTensor of shape (...,) indicating the index of the positive edges to select
        :return: LongTensor of shape (..., 1 + num_negatives, 2) where the positives are located in [:,0,:]
        """
        batch_instances, batch_labels = self.instances[idxs], self.labels[idxs]
        return batch_instances.to(self.device), batch_labels.to(self.device)

    def __len__(self):
        return len(self.instances)

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.instances = self.instances.to(device)
        self.labels = self.labels.to(device)
        return self
