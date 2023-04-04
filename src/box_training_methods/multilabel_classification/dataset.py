import math
from pathlib import Path
from time import time
import pickle
import ijson
from itertools import cycle, islice
from typing import *

import attr
import numpy as np
import pandas as pd
import torch
from loguru import logger
from wcmatch import glob
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, IterableDataset, DataLoader

from skmultilearn.dataset import load_from_arff
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer

__all__ = [
    "edges_from_hierarchy_edge_list",
    "ARFFReader",
    "InstanceLabelsDataset",
]


def edges_from_hierarchy_edge_list(edge_file: Union[Path, str]) -> Tuple[LongTensor, LabelEncoder]:
    """
    Loads edges from a given tsv file into a PyTorch LongTensor.
    Meant for importing data where each edge appears as a line in the file, with
        <child_id>\t<parent_id>\t{}

    :param edge_file: Path of dataset's hierarchy{_tc}.edge_list
    :returns: PyTorch LongTensor of edges with shape (num_edges, 2), LabelEncoder that numerized labels
    """
    start = time()
    logger.info(f"Loading edges from {edge_file}...")
    edges = pd.read_csv(edge_file, sep=" ", header=None).to_numpy()[:, :2]  # ignore line-final "{}"
    # edges[:, [0, 1]] = edges[:, [1, 0]]  # (child, parent) -> (parent, child)
    le = LabelEncoder()
    edges = torch.tensor(le.fit_transform(edges.flatten()).reshape((-1,2)))
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges, le


# https://github.com/iesl/box-mlc-iclr-2022/blob/main/box_mlc/dataset_readers/arff_reader.py
class ARFFReader(object):
    """
    Reader for multilabel datasets in MULAN/WEKA/MEKA datasets.
    This reader supports reading multiple folds kept in separate files. This is done
    by taking in a glob pattern instread of single path.
    For example ::
            '.data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2).arff'
        will match .data/bibtex_stratified10folds_meka/Bibtex-fold1.arff and  .data/bibtex_stratified10folds_meka/Bibtex-fold2.arff
    """

    def __init__(
        self,
        num_labels: int,
        labels_to_skip: Optional[List]=None,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            num_labels: Total number of labels for the dataset.
                Make sure that this is correct. If this is incorrect, the code will not throw error but
                will have a silent bug.
            labels_to_skip: Some HMLC datasets remove the root nodes from the data. These can be specified here.
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_
        """
        super().__init__(**kwargs)
        self.num_labels = num_labels
        if labels_to_skip is None:
            self.labels_to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150'] # hardcode for HMLC datasets for now
            # we can hardcode more labels across datasets as long as they are to be skipped regardless of the dataset
            # because having a label name in this list that is not present in the dataset, won't affect anything.
        else:
            self.labels_to_skip = labels_to_skip

    def read_internal(self, file_path: str) -> List[Dict]:
        """Reads a datafile to produce instances
        Args:
            file_path: glob pattern for files containing folds to read
        Returns:
            List of json containing data examples
        """
        data = []

        for file_ in glob.glob(file_path, flags=glob.EXTGLOB | glob.BRACE):
            logger.info(f"Reading {file_}")
            x, y, feature_names, label_names = load_from_arff(
                file_,
                label_count=self.num_labels,
                return_attribute_definitions=True,
            )
            data += self._arff_dataset(
                x.toarray(), y.toarray(), feature_names, label_names
            )

        return data

    def _arff_dataset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: List[Tuple[str, Any]],
        label_names: List[Tuple[str, Any]],
    ) -> List[Dict]:
        num_features = len(feature_names)
        assert x.shape[-1] == num_features
        num_total_labels = len(label_names)
        assert y.shape[-1] == num_total_labels
        all_labels = np.array([l_[0] for l_ in label_names])
        # remove root
        to_take = np.logical_not(np.in1d(all_labels, self.labels_to_skip)) # shape =(num_labels,),
        # where i = False iff we have to skip
        all_labels = all_labels[to_take]
        data = [
            {
                "x": xi.tolist(),
                "labels": (all_labels[yi[to_take] == 1]).tolist(),
                "idx": str(i),
            }
            for i, (xi, yi) in enumerate(zip(x, y))
            if any(yi)  # skip ex with empty label set
        ]

        return data


@attr.s(auto_attribs=True)
class InstanceLabelsDataset(Dataset):
    """
    """

    instance_feats: Tensor
    labels: Tensor
    label_encoder: LabelEncoder  # label set accessable via label_encoder.classes_

    def __attrs_post_init__(self):

        self._device = self.instance_feats.device
        self.instance_dim = self.instance_feats.shape[1]
        self.labels = self.prune_and_encode_labels_for_instances()
        
        instance_label_pairs = []
        for i, ls in enumerate(self.labels):
            instance_label_pairs.extend([i, l] for l in ls)
        self.instance_label_pairs = torch.tensor(instance_label_pairs)
        
        self.instance_feats = torch.nn.Embedding.from_pretrained(self.instance_feats, freeze=True)

    def __getitem__(self, idxs: LongTensor) -> LongTensor:
        """
        :param idxs: LongTensor of shape (...,) indicating the index of the examples which to select
        :return: batch_instances of shape (batch_size, instance_dim), batch_labels of shape (batch_size, num_labels)
        """
        instance_idxs = self.instance_label_pairs[idxs][:, 0].to(self._device)
        label_idxs = self.instance_label_pairs[idxs][:, 1].to(self._device)
        instance_feats = self.instance_feats(instance_idxs)
        return instance_feats, label_idxs

    def __len__(self):
        return len(self.labels)

    def prune_and_encode_labels_for_instances(self):
        pruned_labels = []
        for ls in self.labels:
            pruned_labels.append(self.label_encoder.transform(self.prune_labels_for_instance(ls)))
        return pruned_labels

    def prune_labels_for_instance(self, ls):
        """only retains most granular labels"""
        pruned_ls = []
        for i in range(len(ls)):
            label_i_is_nobodys_parent = True
            if i < len(ls) - 1:
                for j in range(i+1, len(ls)):
                    if f".{ls[j]}.".startswith(f".{ls[i]}."):
                        label_i_is_nobodys_parent = False
                        break
            if label_i_is_nobodys_parent:
                pruned_ls.append(ls[i])
        return pruned_ls

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.instance_feats = self.instance_feats.to(device)
        # self.labels = self.labels.to(device)
        return self


def collate_mesh_fn(batch, tokenizer):

    inputs = tokenizer([x['journal'] + f' {tokenizer.sep_token} ' + 
                        x['title'] + f' {tokenizer.sep_token} ' + 
                        x['abstractText'] for x in batch], 
                        return_tensors="pt", padding=True)
    labels = [[m for m in x['meshMajor']] for x in batch]

    return inputs, labels


@attr.s(auto_attribs=True)
class BioASQInstanceLabelsDataset(IterableDataset):
    
    file_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/allMesh_2022.json"

    def __attrs_post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

    def parse_file(self, file_path):
        with open(file_path, encoding='windows-1252', mode='r') as f:
            for article in ijson.items(f, 'articles.item'):
                yield article

    def get_stream(self, file_path):
        return cycle(self.parse_file(file_path))

    def __iter__(self):
        return self.get_stream(self.file_path)


# if __name__ == '__main__':
#     d = BioASQInstanceLabelsDataset("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/allMesh_2022.json")
#     dl = DataLoader(d, batch_size=4, collate_fn=lambda b: collate_mesh_fn(batch=b, tokenizer=d.tokenizer))
#     print(next(iter(dl)))
