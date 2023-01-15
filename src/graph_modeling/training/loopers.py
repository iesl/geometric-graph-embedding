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

from __future__ import annotations

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
from pytorch_utils.training import IntervalConditional, ModelCheckpoint
from pytorch_utils.generic import LapTimer
from .metrics import *

__all__ = [
    "TrainLooper",
    "EvalLooper",
]


@attr.s(auto_attribs=True)
class TrainLooper:
    name: str
    model: Module
    dl: DataLoader
    opt: torch.optim.Optimizer
    loss_func: Callable
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    eval_loopers: Iterable[EvalLooper] = attr.ib(factory=tuple)
    early_stopping: Callable = lambda z: None
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[[Dict], Any] = lambda z: None
    log_interval: Optional[Union[IntervalConditional, int]] = attr.ib(
        default=None, converter=IntervalConditional.interval_conditional_converter
    )

    def __attrs_post_init__(self):
        if isinstance(self.eval_loopers, EvalLooper):
            self._eval_loopers = (self.eval_loopers,)
        self.looper_metrics = {"Total Examples": 0}
        if self.log_interval is None:
            # by default, log every batch
            self.log_interval = IntervalConditional(0)
        self.running_losses = []
        self.save_model = ModelCheckpoint()

        self.best_metrics_comparison_functions = {"Mean Loss": min}
        self.best_metrics = {}
        self.previous_best = None

    def loop(self, epochs: int):
        try:
            self.running_losses = []
            for epoch in trange(epochs, desc=f"[{self.name}] Epochs"):
                self.model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)
        except StopLoopingException as e:
            logger.warning(str(e))
        finally:
            self.logger.commit()

            # load in the best model
            previous_device = next(iter(self.model.parameters())).device
            self.model.load_state_dict(self.save_model.best_model_state_dict())
            self.model.to(previous_device)

            # evaluate
            metrics = []
            predictions_coo = []
            for eval_looper in self.eval_loopers:
                metric, prediction_coo = eval_looper.loop()
                metrics.append(metric)
                predictions_coo.append(prediction_coo)
            return metrics, predictions_coo

    def train_loop(self, epoch: Optional[int] = None):
        """
        Internal loop for a single epoch of training
        :return: list of losses per batch
        """
        examples_this_epoch = 0
        if isinstance(self.dl.dataset, Sized):
            examples_in_single_epoch = len(self.dl.dataset)
        else:
            raise NotImplementedError(
                "TrainLooper currently requires datasets to implement `__len__`."
            )
        lap_timer = LapTimer()
        num_batches_since_log = 0
        for iteration, batch_in in enumerate(
            tqdm(self.dl, desc=f"[{self.name}] Batch", leave=False)
        ):
            self.opt.zero_grad()

            batch_out = self.model(batch_in)
            loss = self.loss_func(batch_out)

            # This is not always going to be the right thing to check.
            # In a more general setting, we might want to consider wrapping the DataLoader in some way
            # with something which stores this information.
            num_in_batch = len(loss)

            loss = loss.sum(dim=0)

            self.looper_metrics["Total Examples"] += num_in_batch
            examples_this_epoch += num_in_batch

            if torch.isnan(loss).any():
                raise StopLoopingException("NaNs in loss")
            self.running_losses.append(loss.detach().item())
            loss.backward()

            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        raise StopLoopingException("NaNs in grad")

            num_batches_since_log += 1
            # TODO: Refactor the following
            self.opt.step()
            # If you have a scheduler, keep track of the learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                if len(self.opt.param_groups) == 1:
                    self.looper_metrics[f"Learning Rate"] = self.opt.param_groups[0][
                        "lr"
                    ]
                else:
                    for i, param_group in enumerate(self.opt.param_groups):
                        self.looper_metrics[f"Learning Rate (Group {i})"] = param_group[
                            "lr"
                        ]

            # Check performance every self.log_interval number of examples
            last_log = self.log_interval.last

            if self.log_interval(self.looper_metrics["Total Examples"]):
                average_time_per_batch = lap_timer.elapsed() / num_batches_since_log
                self.logger.collect({"Average Time Per Batch": average_time_per_batch})
                num_batches_since_log = 0

                self.logger.collect(self.looper_metrics)
                mean_loss = sum(self.running_losses) / (
                    self.looper_metrics["Total Examples"] - last_log
                )
                metrics = {"Mean Loss": mean_loss}
                self.logger.collect(
                    {
                        **{
                            f"[{self.name}] {metric_name}": value
                            for metric_name, value in metrics.items()
                        },
                        "Epoch": epoch + examples_this_epoch / examples_in_single_epoch,
                    }
                )
                self.logger.commit()
                self.running_losses = []
                self.update_best_metrics_(metrics)
                self.save_if_best_(self.best_metrics["Mean Loss"])
                self.early_stopping(self.best_metrics["Mean Loss"])

    def update_best_metrics_(self, metrics: Dict[str, float]) -> None:
        for name, comparison in self.best_metrics_comparison_functions.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = metrics[name]
            else:
                self.best_metrics[name] = comparison(
                    metrics[name], self.best_metrics[name]
                )
        self.summary_func(
            {
                f"[{self.name}] Best {name}": val
                for name, val in self.best_metrics.items()
            }
        )

    def save_if_best_(self, best_metric) -> None:
        if best_metric != self.previous_best:
            self.save_model(self.model)
            self.previous_best = best_metric


@attr.s(auto_attribs=True)
class EvalLooper:
    """
    Run an evaluation loop on the full adjacency matrix.
    The evaluation is performed on GPU in batches.
    *Note:* To maximize available RAM we also *unload* the dataset from GPU;
    Therefore, this eval loop should only be performed one time at the end,
    after all training is completed.
    """

    name: str
    model: Module
    dl: DataLoader
    batch_size: int
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[[Dict], Any] = lambda z: None

    @torch.no_grad()
    def loop(self) -> Tuple[Dict[str, Any], coo_matrix]:
        self.model.eval()

        logger.debug("Evaluating model predictions on full adjacency matrix")
        lap_timer = LapTimer()
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
                cur_pos += self.batch_size
                if cur_pos > number_of_entries:
                    cur_pos = number_of_entries

                ids = torch.tensor(input_list[last_pos:cur_pos], dtype=torch.long)
                cur_preds = self.model(ids.to(previous_device)).cpu().numpy()
                prediction_scores[
                    input_x[last_pos:cur_pos], input_y[last_pos:cur_pos]
                ] = cur_preds
                pbar.update(self.batch_size)

        prediction_scores_no_diag = prediction_scores[~np.eye(num_nodes, dtype=bool)]
        ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]

        logger.debug(f"Evaluation time: {lap_timer.elapsed()}")

        del input_x, input_y

        logger.debug("Calculating optimal F1 score")
        metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)
        logger.debug(f"F1 calculation time: {lap_timer.elapsed()}")
        logger.info(f"Metrics: {metrics}")

        self.logger.collect({f"[{self.name}] {k}": v for k, v in metrics.items()})
        self.logger.commit()

        predictions = (prediction_scores > metrics["threshold"]) * (
            ~np.eye(num_nodes, dtype=bool)
        )

        return metrics, coo_matrix(predictions)
