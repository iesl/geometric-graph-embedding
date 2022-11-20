import os
from .arff_reader import ARFFReader


def generate(indir, num_labels=500, **kwargs):

    reader = ARFFReader(num_labels=num_labels)
    data_train = list(reader.read_internal(os.path.join(indir, "train-normalized.arff")))
    data_dev = list(reader.read_internal(os.path.join(indir, "dev-normalized.arff")))
    data_test = list(reader.read_internal(os.path.join(indir, "test-normalized.arff")))

    return data_train, data_dev, data_test
