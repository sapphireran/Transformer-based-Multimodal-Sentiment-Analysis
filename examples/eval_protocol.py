"""MOSI/MOSEI-style metrics, copied from ``model/train_and_test.py``.

Kept here so examples do not import ``train_and_test`` (that module pulls
``memory_profiler`` and calls ``plt.show()`` inside ``single_test``).

Formulas: ``docs/metrics.md``.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

Array = Union[np.ndarray, "torch.Tensor"]


def _as_numpy(x: Array) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def split_uniform_7(data) -> np.ndarray:
    """Map [-3, 3] to bins 1..7 with equal width 6/7."""
    data = _as_numpy(data)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data) -> np.ndarray:
    """Map [-3, 3] to bins 1..5 with equal width 6/5."""
    data = _as_numpy(data)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 5)


def acc7_edges() -> list:
    step = 6.0 / 7.0
    return [-3.0 + i * step for i in range(8)]


def acc5_edges() -> list:
    step = 6.0 / 5.0
    return [-3.0 + i * step for i in range(6)]


def eval_affect(truths, results, exclude_zero: bool = True):
    """Binary F1 / Acc2 on the non-zero gold subset (default)."""
    test_preds = _as_numpy(results)
    test_truth = _as_numpy(truths)
    keep = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )
    if keep.size == 0:
        return 0.0, 0.0
    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0
    f1 = f1_score(binary_truth, binary_preds, average="binary", zero_division=0)
    accuracy = accuracy_score(binary_truth, binary_preds)
    return float(f1), float(accuracy)


def regression_and_classification(truths, predictions) -> Dict[str, float]:
    """Full protocol used by ``single_test`` minus the confusion-matrix plot."""
    y = _as_numpy(truths).astype(np.float64)
    yhat = _as_numpy(predictions).astype(np.float64)
    if y.size == 0:
        raise ValueError("empty arrays")

    mae = float(np.mean(np.abs(y - yhat)))
    mse = float(np.mean((y - yhat) ** 2))
    if y.size >= 2 and np.std(y) > 0 and np.std(yhat) > 0:
        corr, _ = pearsonr(y, yhat)
        corr = float(corr)
    else:
        corr = float("nan")

    acc7 = float(accuracy_score(split_uniform_7(y), split_uniform_7(yhat)))
    acc5 = float(accuracy_score(split_uniform_5(y), split_uniform_5(yhat)))
    f1, acc2 = eval_affect(y, yhat, exclude_zero=True)
    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": corr,
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def format_metrics(metrics: Dict[str, float], digits: int = 4) -> str:
    order = ["MAE", "MSE", "Corr", "Acc7_uniform", "Acc5_uniform", "Acc2", "F1"]
    parts = []
    for key in order:
        if key not in metrics:
            continue
        value = metrics[key]
        parts.append(f"{key}={value:.{digits}f}" if value == value else f"{key}=nan")
    return "  ".join(parts)
