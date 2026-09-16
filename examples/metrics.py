"""Affect metrics copied from ``model/train_and_test.py`` without plotting.

``single_test`` also builds a matplotlib confusion matrix and calls
``plt.show()``. These helpers keep the numeric definitions only so the
examples can run headless.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score


def split_uniform(data, n_bins: int) -> np.ndarray:
    """Map values in [-3, 3] onto ``{1, …, n_bins}`` with equal-width bins."""
    data = np.asarray(data).reshape(-1)
    step = 6.0 / n_bins
    edges = [-3.0 + i * step for i in range(n_bins + 1)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, n_bins)


def split_uniform_7(data) -> np.ndarray:
    return split_uniform(data, 7)


def split_uniform_5(data) -> np.ndarray:
    return split_uniform(data, 5)


def eval_affect(truths, results, exclude_zero: bool = True) -> tuple[float, float]:
    """Binary F1 and accuracy after dropping gold zeros (CMU affect protocol)."""
    pred = np.asarray(results).reshape(-1)
    truth = np.asarray(truths).reshape(-1)
    keep = np.array([i for i, value in enumerate(truth) if value != 0 or not exclude_zero])
    binary_truth = truth[keep] > 0
    binary_pred = pred[keep] > 0
    return (
        float(f1_score(binary_truth, binary_pred, average="binary")),
        float(accuracy_score(binary_truth, binary_pred)),
    )


def regression_and_bins(truths, results) -> dict[str, float]:
    """MAE / MSE / Pearson r plus Acc-7 / Acc-5 / Acc-2 / F1."""
    pred = np.asarray(results, dtype=np.float64).reshape(-1)
    truth = np.asarray(truths, dtype=np.float64).reshape(-1)
    if np.std(pred) < 1e-12 or np.std(truth) < 1e-12:
        corr = float("nan")
    else:
        corr, _ = pearsonr(truth, pred)
    mse = float(np.mean((truth - pred) ** 2))
    mae = float(np.mean(np.abs(truth - pred)))
    acc7 = float(accuracy_score(split_uniform_7(truth), split_uniform_7(pred)))
    acc5 = float(accuracy_score(split_uniform_5(truth), split_uniform_5(pred)))
    f1, acc2 = eval_affect(truth, pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": float(corr),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def bin_edges(n_bins: int) -> list[float]:
    step = 6.0 / n_bins
    return [-3.0 + i * step for i in range(n_bins + 1)]
