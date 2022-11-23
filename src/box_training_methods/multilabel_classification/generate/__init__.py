from importlib import import_module
from pathlib import Path
from pprint import pformat
from typing import *

import toml
from loguru import logger
import pickle

__all__ = [
    "write_dataset",
]


def write_dataset(out_dir: Union[str, Path], **mlc_config):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mlc_module = import_module("multilabel_classification.generate.mlc")

    logger.info("Generating multilabel dataset with the following config:\n" + pformat(mlc_config))
    data_train, data_dev, data_test = mlc_module.generate(**mlc_config)

    mlc_sub_configs = []
    for name in sorted(mlc_config.keys()):
        if name not in ['indir']:
            mlc_sub_configs.append(f"{name}={mlc_config[name]}")
    mlc_folder_name = "-".join(mlc_sub_configs)
    mlc_folder = out_dir / f"{mlc_folder_name}/"
    mlc_folder.mkdir(parents=True, exist_ok=True)
    mlc_file_stub = mlc_folder / mlc_config["dataset_name"]
    logger.info(f"Saving to {mlc_file_stub}")
    with mlc_file_stub.with_suffix(".pkl").open("wb") as f_pkl:
        pickle.dump({
            "train": data_train,
            "dev": data_dev,
            "test": data_test,
        }, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    with mlc_file_stub.with_suffix(".toml").open("w") as f_toml:
        toml.dump({
            "dataset_name": mlc_config["dataset_name"],
            "num_labels": mlc_config["num_labels"],
        }, f_toml)
