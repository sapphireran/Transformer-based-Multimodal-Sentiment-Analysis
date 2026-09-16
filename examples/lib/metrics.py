"""Metric helpers that mirror `model/train_and_test.py` without matplotlib.

Keep this file aligned with `docs/evaluation.md` and the original
`split_uniform_*` / `eval_affect` functions. Tests compare the two
implementations when the training module can be imported.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

ArrayLike = Union[np.ndarray, Iterable[float]]


def _as_1d(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("expected a non-empty array")
    return arr


def split_uniform_7(data: ArrayLike) -> np.ndarray:
    """Map values on [-3, 3] to equal-width bins 1..7 (same edges as training)."""
    data = _as_1d(data)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data: ArrayLike) -> np.ndarray:
    """Map values on [-3, 3] to equal-width bins 1..5."""
    data = _as_1d(data)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 5)


def uniform_edges(n_bins: int) -> Tuple[float, ...]:
    if n_bins not in (5, 7):
        raise ValueError("n_bins must be 5 or 7")
    step = 6.0 / n_bins
    return tuple(-3.0 + i * step for i in range(n_bins + 1))


def binary_acc_f1(
    truths: ArrayLike,
    preds: ArrayLike,
    exclude_zero: bool = True,
) -> Tuple[float, float]:
    """Same rule as `eval_affect`: drop y==0, then (value > 0). Returns (f1, acc)."""
    y = _as_1d(truths)
    yhat = _as_1d(preds)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: truths {y.shape} vs preds {yhat.shape}")

    keep = np.ones(y.shape[0], dtype=bool)
    if exclude_zero:
        keep = y != 0.0
    if not np.any(keep):
        raise ValueError("no samples left after exclude_zero")

    binary_truth = y[keep] > 0
    binary_preds = yhat[keep] > 0
    f1 = float(f1_score(binary_truth, binary_preds, average="binary"))
    acc = float(accuracy_score(binary_truth, binary_preds))
    return f1, acc


def regression_metrics(truths: ArrayLike, preds: ArrayLike) -> Dict[str, float]:
    y = _as_1d(truths)
    yhat = _as_1d(preds)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: truths {y.shape} vs preds {yhat.shape}")
    mae = float(np.mean(np.abs(yhat - y)))
    mse = float(np.mean((yhat - y) ** 2))
    if np.std(y) == 0.0 or np.std(yhat) == 0.0:
        corr = float("nan")
    else:
        corr, _ = pearsonr(y, yhat)
        corr = float(corr)
    return {"MAE": mae, "MSE": mse, "Corr": corr}


def summarize_predictions(
    truths: ArrayLike,
    preds: ArrayLike,
    exclude_zero: bool = True,
) -> Dict[str, float]:
    """Full `single_test` metric set except the training criterion / plots."""
    y = _as_1d(truths)
    yhat = _as_1d(preds)
    out = regression_metrics(y, yhat)
    acc7 = float(accuracy_score(split_uniform_7(y), split_uniform_7(yhat)))
    acc5 = float(accuracy_score(split_uniform_5(y), split_uniform_5(yhat)))
    f1, acc2 = binary_acc_f1(y, yhat, exclude_zero=exclude_zero)
    out.update(
        {
            "Acc7_uniform": acc7,
            "Acc5_uniform": acc5,
            "Acc2": acc2,
            "F1": f1,
        }
    )
    return out
