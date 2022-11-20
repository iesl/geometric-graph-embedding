"""
Copied from Dhruvesh Patel's:
https://github.com/iesl/box-mlc-iclr-2022/blob/main/box_mlc/dataset_readers/arff_reader.py
"""

"""Dataset reader for multilabel dataset in arff (MEKA) format.
 See `this <http://www.uco.es/kdis/mllresources>`_ for reference."""

from typing import (
    Dict,
    List,
    Any,
    Iterator,
    cast,
    Tuple,
    Iterable,
    Optional,
)
import sys
import logging
import json
import numpy as np
from skmultilearn.dataset import load_from_arff
import torch
from wcmatch import glob

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


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
