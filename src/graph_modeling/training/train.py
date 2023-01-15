# Copyright 2021 The Geometric Graph Embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import random
import uuid
from pathlib import Path
from time import time
from typing import *

import scipy.sparse
import toml
import torch
import wandb
from loguru import logger
from torch.nn import Module
from wandb_utils.loggers import WandBLogger

from graph_modeling.models.temps import (
    GlobalTemp,
    PerDimTemp,
    PerEntityTemp,
    PerEntityPerDimTemp,
)
from pytorch_utils import TensorDataLoader, cuda_if_available
from pytorch_utils.training import EarlyStopping, ModelCheckpoint
from .dataset import (
    edges_from_tsv,
    edges_and_num_nodes_from_npz,
    RandomNegativeEdges,
    GraphDataset,
)
from .loopers import TrainLooper, EvalLooper
from .loss import (
    BCEWithLogsNegativeSamplingLoss,
    BCEWithLogitsNegativeSamplingLoss,
    BCEWithDistancesNegativeSamplingLoss,
    MaxMarginOENegativeSamplingLoss,
)
from .. import metric_logger
from ..models.box import BoxMinDeltaSoftplus, TBox
from ..models.hyperbolic import (
    Lorentzian,
    LorentzianDistance,
    LorentzianScore,
    HyperbolicEntailmentCones,
)
from ..models.poe import OE, POE
from ..models.vector import VectorSim, VectorDist, BilinearVector, ComplexVector

__all__ = [
    "training",
    "setup_training_data",
    "setup_model",
    "setup",
]


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

    graph = Path(config["data_path"])
    if graph.is_dir():
        # If graph is set to a directory, we select a graph from that directory uniformly randomly
        graphs = list({file.stem for file in graph.glob("*.npz")})
        logger.info(f"Directory {graph} has {len(graphs)} graph files")
        selected_graph_name = random.choice(graphs)
        logger.info(f"Selected graph {selected_graph_name}")
        config["data_path"] = str(graph / selected_graph_name)

    if config["undirected"] is None:
        config["undirected"] = config["model_type"] == "lorentzian"
        logger.debug(
            f"Setting undirected={config['undirected']} since model_type={config['model_type']}"
        )

    if config["wandb"]:
        wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
    else:
        run_dir = Path(".")

    dataset, dataloader, model, train_looper = setup(**config)
    train_looper.save_model.run_dir = run_dir

    if config["wandb"]:
        metric_logger.metric_logger = WandBLogger()
        wandb.watch(model)
        for eval_looper in train_looper.eval_loopers:
            eval_looper.summary_func = wandb.run.summary.update

    # TODO: refactor so train looper simply imports metric_logger
    train_looper.logger = metric_logger.metric_logger
    for eval_looper in train_looper.eval_loopers:
        eval_looper.logger = train_looper.logger

    metrics, predictions_coo = train_looper.loop(config["epochs"])

    # saving output results
    if config["output_dir"] == None:
        output_parent_dir = Path(os.path.dirname(config["data_path"])) / "results"
    else:
        output_parent_dir = Path(config["output_dir"])
    model_string = config["model_type"]
    if model_string == "tbox":
        model_string += f"_{config['tbox_temperature_type']}"
    model_string += f"_{config['dim']}"
    output_dir = output_parent_dir / model_string
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_id = os.path.basename(config["data_path"])
    random_hex = wandb.run.id if config["wandb"] else uuid.uuid4().hex
    with open(output_dir / f"{graph_id}_{random_hex}.metric", "w") as f:
        f.write(json.dumps(dict(config)))
        f.write("\n")
        f.write(json.dumps(metrics))

    if config["save_model"]:
        train_looper.save_model.save_to_disk(None)

    if config["save_prediction"]:
        if len(predictions_coo) > 0 and predictions_coo[0] is not None:
            filename_pred = f"{output_dir}/{graph_id}_{random_hex}.prediction"
            scipy.sparse.save_npz(filename_pred, predictions_coo[0])  # check this part
        else:
            raise ValueError(
                "save_prediction was requested, but no predictions returned from training loop"
            )

    if config["wandb"]:
        wandb.finish()

    logger.info("Training complete!")


def setup_model(
    num_nodes: int, device: Union[str, torch.device], **config
) -> Tuple[Module, Callable]:
    # TODO: Break this out into the model directory
    model_type = config["model_type"].lower()
    if model_type == "gumbel_box":
        model = BoxMinDeltaSoftplus(
            num_nodes,
            config["dim"],
            volume_temp=config["box_volume_temp"],
            intersection_temp=config["box_intersection_temp"],
        )
        loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "tbox":
        temp_type = {
            "global": GlobalTemp,
            "per_dim": PerDimTemp,
            "per_entity": PerEntityTemp,
            "per_entity_per_dim": PerEntityPerDimTemp,
        }
        Temp = temp_type[config["tbox_temperature_type"]]

        model = TBox(
            num_nodes,
            config["dim"],
            intersection_temp=Temp(
                config["box_intersection_temp"],
                0.0001,
                100,
                dim=config["dim"],
                num_entities=num_nodes,
            ),
            volume_temp=Temp(
                config["box_volume_temp"],
                0.01,
                1000,
                dim=config["dim"],
                num_entities=num_nodes,
            ),
        )
        loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "order_embeddings":
        model = OE(num_nodes, config["dim"])
        loss_func = MaxMarginOENegativeSamplingLoss(
            config["negative_weight"], config["margin"]
        )
    elif model_type == "partial_order_embeddings":
        model = POE(num_nodes, config["dim"])
        loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "vector_sim":
        model = VectorSim(
            num_nodes,
            config["dim"],
            config["vector_separate_io"],
            config["vector_use_bias"],
        )
        loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "vector_dist":
        model = VectorDist(num_nodes, config["dim"], config["vector_separate_io"],)
        loss_func = BCEWithDistancesNegativeSamplingLoss(
            config["negative_weight"], config["margin"],
        )
    elif model_type == "bilinear_vector":
        model = BilinearVector(
            num_nodes,
            config["dim"],
            config["vector_separate_io"],
            config["vector_use_bias"],
        )
        loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "complex_vector":
        model = ComplexVector(num_nodes, config["dim"],)
        loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "lorentzian":
        model = Lorentzian(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        loss_func = BCEWithDistancesNegativeSamplingLoss(config["negative_weight"])
        # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "lorentzian_score":
        model = LorentzianScore(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
        # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "lorentzian_distance":
        model = LorentzianDistance(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
        # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "hyperbolic_entailment_cones":
        model = HyperbolicEntailmentCones(
            num_nodes,
            config["dim"],
            config["hyperbolic_entailment_cones_relative_cone_aperture_scale"],
            config["hyperbolic_entailment_cones_eps_bound"],
        )
        loss_func = MaxMarginOENegativeSamplingLoss(
            config["negative_weight"], config["margin"]
        )
    else:
        raise ValueError(f"Model type {config['model_type']} does not exist")
    model.to(device)

    return model, loss_func


def setup_training_data(device: Union[str, torch.device], **config) -> GraphDataset:
    """
    Load the training data (either npz or tsv)

    :param device: device to load training data on
    :param config: config dictionary

    :returns: GraphDataset with appropriate negative sampling, ready for training
    """
    start = time()
    npz_file = Path(config["data_path"] + ".npz")
    tsv_file = Path(config["data_path"] + ".tsv")
    avoid_edges = None
    if npz_file.exists():
        training_edges, num_nodes = edges_and_num_nodes_from_npz(npz_file)
    elif tsv_file.exists():
        stats = toml.load(config["data_path"] + ".toml")
        num_nodes = stats["num_nodes"]
        training_edges = edges_from_tsv(tsv_file)
        avoid_file = Path(config["data_path"] + ".avoid.tsv")
        if avoid_file.exists():
            avoid_edges = edges_from_tsv(avoid_file)
            logger.debug(f"Loaded {len(avoid_edges)} edges to avoid from {avoid_file}")
    else:
        raise ValueError(
            f"Could not locate training file at {config['data_path']}{{.npz,.tsv}}"
        )
    training_edges = training_edges.to(device)
    if config["undirected"]:
        training_edges = torch.unique(torch.sort(training_edges, dim=-1).values, dim=0)
    if avoid_edges is None:
        diag = torch.arange(num_nodes, device=device)[:, None].expand(-1, 2)
        if config["undirected"]:
            # The following is not particularly memory efficient, but should serve our purpose
            avoid_edges = torch.cat((training_edges, training_edges[..., [1, 0]], diag))
        else:
            avoid_edges = torch.cat((training_edges, diag))

    negative_sampler = RandomNegativeEdges(
        num_nodes=num_nodes,
        negative_ratio=config["negative_ratio"],
        avoid_edges=avoid_edges,
        device=device,
        permutation_option=config["negatives_permutation_option"],
    )

    dataset = GraphDataset(
        training_edges, num_nodes=num_nodes, negative_sampler=negative_sampler
    )

    logger.info(f"Number of edges in dataset: {dataset.num_edges:,}")
    logger.info(f"Number of edges to avoid: {len(avoid_edges):,}")
    logger.info(
        f"Number of negative edges: {num_nodes * (num_nodes - 1) - dataset.num_edges:,}"
    )
    logger.info(f"Density: {100*dataset.num_edges / (num_nodes * (num_nodes -1)):5f}%")
    logger.debug(f"Total time spent loading data: {time()-start:0.1f} seconds")

    return dataset


def setup(**config):
    """
    Setup and return the datasets, dataloaders, model, and training loop required for training.

    :param config: config dictionary
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """
    device = cuda_if_available(use_cuda=config["cuda"])

    # setup data
    train_dataset = setup_training_data(device, **config)
    dataloader = TensorDataLoader(
        train_dataset, batch_size=2 ** config["log_batch_size"], shuffle=True
    )

    if isinstance(config["log_interval"], float):
        config["log_interval"] = math.ceil(len(train_dataset) * config["log_interval"])
    logger.info(f"Log every {config['log_interval']:,} instances")
    logger.info(f"Stop after {config['patience']:,} logs show no improvement in loss")

    # setup model
    model, loss_func = setup_model(train_dataset.num_nodes, device, **config)

    # setup optimizer
    opt = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.0
    )

    # set Eval Looper
    eval_loopers = []
    if config["eval"]:
        logger.debug(f"After training, will evaluate on full adjacency matrix")
        eval_loopers.append(
            EvalLooper(
                name="Train",  # this is used for logging to describe the dataset, which is the same data as in train
                model=model,
                dl=dataloader,
                batch_size=2 ** config["log_eval_batch_size"],
            )
        )
    train_looper = TrainLooper(
        name="Train",
        model=model,
        dl=dataloader,
        opt=opt,
        loss_func=loss_func,
        eval_loopers=eval_loopers,
        log_interval=config["log_interval"],
        early_stopping=EarlyStopping("Loss", config["patience"]),
    )

    return train_dataset, dataloader, model, train_looper
