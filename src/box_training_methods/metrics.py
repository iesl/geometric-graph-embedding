from typing import *

import numpy as np
import sklearn.metrics
from loguru import logger

__all__ = [
    "calculate_optimal_threshold",
    "calculate_optimal_F1_threshold",
    "calculate_metrics",
    "numpy_metrics",
    "calculate_optimal_F1",
]


def calculate_optimal_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        targets, scores, drop_intermediate=False
    )
    return thresholds[np.argmax(tpr - fpr)]


def calculate_optimal_F1_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        targets, scores, drop_intermediate=False
    )
    num_pos = targets.sum()
    num_neg = (1 - targets).sum()
    logger.debug(f"Calculating F1 with {num_pos} positive, {num_neg} negative")
    f1 = 2 * tpr / (1 + tpr + fpr * num_neg / num_pos)
    return thresholds[np.argmax(f1)]


def calculate_metrics(
    targets: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    scores_hard = scores > threshold
    return {
        "Accuracy": (scores_hard == targets).mean(),
        "F1": sklearn.metrics.f1_score(targets, scores_hard),
    }


def numpy_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    assert targets.dtype == np.bool
    assert predictions.dtype == np.bool

    true_positives = (predictions & targets).sum()
    false_positives = (predictions & ~targets).sum()
    false_negatives = (~predictions & targets).sum()
    return {
        "Accuracy": (predictions == targets).mean(),
        "F1": true_positives
        / (true_positives + (false_positives + false_negatives) / 2),
    }


def calculate_optimal_F1(targets, scores) -> Dict[str, float]:
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        targets, scores, drop_intermediate=False
    )
    auc = sklearn.metrics.auc(fpr, tpr)
    num_pos = targets.sum()
    num_neg = (1 - targets).sum()
    f1 = 2 * tpr / (1 + tpr + fpr * num_neg / num_pos)
    threshold = thresholds[np.argmax(f1)]
    return {"F1": float(np.max(f1)), "AUC": float(auc), "threshold": threshold}
