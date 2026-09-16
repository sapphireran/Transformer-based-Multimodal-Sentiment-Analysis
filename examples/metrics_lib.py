"""Affect metrics that match ``model/train_and_test.py`` without importing it.

``train_and_test`` pulls in ``memory_profiler`` and calls ``plt.show()`` inside
``single_test``. Examples and tests use this module instead. Formulas are the
ones documented in ``docs/evaluation.md``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def split_uniform_7(data) -> np.ndarray:
    """Map ``[-3, 3]`` onto 7 equal-width bins with ids in ``{1..7}``."""
    data = _as_1d(data)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data) -> np.ndarray:
    """Map ``[-3, 3]`` onto 5 equal-width bins with ids in ``{1..5}``."""
    data = _as_1d(data)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 5)


def uniform_edges(n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    step = 6.0 / n_bins
    return np.asarray([-3.0 + i * step for i in range(n_bins + 1)], dtype=np.float64)


def eval_affect(truths, results, exclude_zero: bool = True) -> tuple[float, float]:
    """Binary F1 and accuracy on ``y > 0``, optionally dropping true zeros.

    Returns ``(f1, accuracy)`` in that order, same as ``train_and_test.eval_affect``.
    """
    test_preds = _as_1d(results)
    test_truth = _as_1d(truths)
    if exclude_zero:
        keep = test_truth != 0
    else:
        keep = np.ones_like(test_truth, dtype=bool)
    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0
    f1 = float(f1_score(binary_truth, binary_preds, average="binary"))
    acc = float(accuracy_score(binary_truth, binary_preds))
    return f1, acc


def evaluate_affect_batch(truths, preds) -> Dict[str, float]:
    """Return the same keys ``single_test`` puts in its result dict (no TestLoss)."""
    y = _as_1d(truths)
    yhat = _as_1d(preds)
    if y.size == 0:
        raise ValueError("empty arrays")
    corr, _ = pearsonr(y, yhat)
    mse = float(np.mean((y - yhat) ** 2))
    mae = float(np.mean(np.abs(y - yhat)))
    acc7 = float(accuracy_score(split_uniform_7(y), split_uniform_7(yhat)))
    acc5 = float(accuracy_score(split_uniform_5(y), split_uniform_5(yhat)))
    f1, acc2 = eval_affect(y, yhat, exclude_zero=True)
    return {
        "MSE": mse,
        "MAE": mae,
        "Corr": float(corr),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def format_metrics(row: Mapping[str, float], digits: int = 4) -> str:
    order = ["MAE", "MSE", "Corr", "Acc7_uniform", "Acc5_uniform", "Acc2", "F1"]
    parts = []
    for key in order:
        if key in row:
            parts.append(f"{key}={row[key]:.{digits}f}")
    return "  ".join(parts)
