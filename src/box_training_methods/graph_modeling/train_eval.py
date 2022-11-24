from __future__ import annotations

import time
from typing import *

import attr
import numpy as np
import torch
from loguru import logger
from scipy.sparse import coo_matrix
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm

from pytorch_utils.exceptions import StopLoopingException
from pytorch_utils.loggers import Logger
from pytorch_utils.training import IntervalConditional
from ..metrics import *

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

from box_training_methods.models.temps import (
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
from .loss import (
    BCEWithLogsNegativeSamplingLoss,
    BCEWithLogitsNegativeSamplingLoss,
    BCEWithDistancesNegativeSamplingLoss,
    MaxMarginOENegativeSamplingLoss,
    PushApartPullTogetherLoss,
)
from box_training_methods.models.box import BoxMinDeltaSoftplus, TBox, HardBox
from box_training_methods.models.hyperbolic import (
    Lorentzian,
    LorentzianDistance,
    LorentzianScore,
    HyperbolicEntailmentCones,
)
from box_training_methods.models.poe import OE, POE
from box_training_methods.models.vector import VectorSim, VectorDist, BilinearVector, ComplexVector
from box_training_methods.train_eval.loopers import EvalLooper

__all__ = [
    "setup_model",
    "setup_training_data",
    "GraphModelingEvalLooper",
]


# TODO make num_nodes a kwarg
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
    elif model_type == "hard_box":
        model = HardBox(
            num_nodes,
            config["dim"],
            constrain_deltas_fn=config["constrain_deltas_fn"]
        )
        loss_func = PushApartPullTogetherLoss(config["negative_weight"])
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

    graph = Path(config["data_path"])
    if graph.is_dir():
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


@attr.s(auto_attribs=True)
class GraphModelingEvalLooper(EvalLooper):

    @torch.no_grad()
    def loop(self) -> Dict[str, Any]:
        self.model.eval()

        logger.debug("Evaluating model predictions on full adjacency matrix")
        time1 = time.time()
        previous_device = next(iter(self.model.parameters())).device
        num_nodes = self.dl.dataset.num_nodes
        ground_truth = np.zeros((num_nodes, num_nodes))
        pos_index = self.dl.dataset.edges.cpu().numpy()
        # release RAM
        del self.dl.dataset

        ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1

        prediction_scores = np.zeros((num_nodes, num_nodes))  # .to(previous_device)

        input_x, input_y = np.indices((num_nodes, num_nodes))
        input_x, input_y = input_x.flatten(), input_y.flatten()
        input_list = np.stack([input_x, input_y], axis=-1)
        number_of_entries = len(input_x)

        with torch.no_grad():
            pbar = tqdm(
                desc=f"[{self.name}] Evaluating", leave=False, total=number_of_entries
            )
            cur_pos = 0
            while cur_pos < number_of_entries:
                last_pos = cur_pos
                cur_pos += self.batchsize
                if cur_pos > number_of_entries:
                    cur_pos = number_of_entries

                ids = torch.tensor(input_list[last_pos:cur_pos], dtype=torch.long)
                cur_preds = self.model(ids.to(previous_device)).cpu().numpy()
                prediction_scores[
                    input_x[last_pos:cur_pos], input_y[last_pos:cur_pos]
                ] = cur_preds
                pbar.update(self.batchsize)

        prediction_scores_no_diag = prediction_scores[~np.eye(num_nodes, dtype=bool)]
        ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]

        time2 = time.time()
        logger.debug(f"Evaluation time: {time2 - time1}")

        # TODO: release self.dl from gpu
        del input_x, input_y

        logger.debug("Calculating optimal F1 score")
        metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)
        time3 = time.time()
        logger.debug(f"F1 calculation time: {time3 - time2}")
        logger.info(f"Metrics: {metrics}")

        self.logger.collect({f"[{self.name}] {k}": v for k, v in metrics.items()})
        self.logger.commit()

        predictions = (prediction_scores > metrics["threshold"]) * (
            ~np.eye(num_nodes, dtype=bool)
        )

        return metrics, coo_matrix(predictions)
