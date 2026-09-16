"""Headless copy of the MOSI / MOSEI metrics in ``train_and_test.py``.

``single_test`` opens a matplotlib confusion-matrix window. This module keeps
the same arithmetic (uniform Acc-5 / Acc-7, drop-zero Acc-2 / F1, Pearson r)
so examples and pytest can run without a display.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score


def split_uniform_7(data) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, 5)


def eval_affect(truths, results, exclude_zero: bool = True) -> tuple[float, float]:
    test_preds = np.asarray(results, dtype=np.float64).reshape(-1)
    test_truth = np.asarray(truths, dtype=np.float64).reshape(-1)
    keep = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)],
        dtype=int,
    )
    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0
    f1 = f1_score(binary_truth, binary_preds, average="binary")
    accuracy = accuracy_score(binary_truth, binary_preds)
    return float(f1), float(accuracy)


def regression_metrics(truths, results) -> dict[str, float]:
    y = np.asarray(truths, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(results, dtype=np.float64).reshape(-1)
    corr, _ = pearsonr(y, y_hat)
    mse = float(np.mean((y - y_hat) ** 2))
    mae = float(np.mean(np.abs(y - y_hat)))
    return {"MAE": mae, "MSE": mse, "Corr": float(corr)}


def evaluate(truths, results) -> dict[str, float]:
    """Return the same keys ``single_test`` puts in its result dict (minus TestLoss)."""
    y = np.asarray(truths, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(results, dtype=np.float64).reshape(-1)
    out = regression_metrics(y, y_hat)
    f1, acc2 = eval_affect(y, y_hat)
    out["Acc7_uniform"] = float(accuracy_score(split_uniform_7(y), split_uniform_7(y_hat)))
    out["Acc5_uniform"] = float(accuracy_score(split_uniform_5(y), split_uniform_5(y_hat)))
    out["Acc2"] = acc2
    out["F1"] = f1
    return out


def round_row(metrics: Mapping[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(float(v), digits) for k, v in metrics.items()}


def bin_edges_7() -> list[float]:
    step = 6.0 / 7.0
    return [-3.0 + i * step for i in range(8)]


def bin_edges_5() -> list[float]:
    step = 6.0 / 5.0
    return [-3.0 + i * step for i in range(6)]
