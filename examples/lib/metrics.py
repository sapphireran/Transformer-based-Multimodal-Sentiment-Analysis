"""NumPy port of the evaluation helpers in model/train_and_test.py.

Kept independent of PyTorch, matplotlib, and memory_profiler so the metric
walkthrough can run in a bare environment. Formulas match single_test /
eval_affect / split_uniform_* as of the current personal tree.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _as_1d(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("expected at least one value")
    return array


def mae(truth: Iterable[float], pred: Iterable[float]) -> float:
    t, p = _as_1d(truth), _as_1d(pred)
    return float(np.mean(np.abs(t - p)))


def mse(truth: Iterable[float], pred: Iterable[float]) -> float:
    t, p = _as_1d(truth), _as_1d(pred)
    return float(np.mean((t - p) ** 2))


def pearson_corr(truth: Iterable[float], pred: Iterable[float]) -> float:
    t, p = _as_1d(truth), _as_1d(pred)
    if t.size < 2:
        return float("nan")
    if np.std(t) == 0.0 or np.std(p) == 0.0:
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def split_uniform_7(data: Iterable[float]) -> np.ndarray:
    """Map values on [-3, 3] to classes 1..7 with equal-width bins."""
    values = _as_1d(data)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data: Iterable[float]) -> np.ndarray:
    """Map values on [-3, 3] to classes 1..5 with equal-width bins."""
    values = _as_1d(data)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, 5)


def acc2_f1(
    truth: Iterable[float], pred: Iterable[float], exclude_zero: bool = True
) -> Tuple[float, float]:
    """Binary F1 and accuracy after dropping exact-zero gold labels.

    Returns (f1, accuracy) in the same order as eval_affect.
    """
    t, p = _as_1d(truth), _as_1d(pred)
    if exclude_zero:
        keep = t != 0.0
        t, p = t[keep], p[keep]
    if t.size == 0:
        return float("nan"), float("nan")

    binary_truth = t > 0.0
    binary_pred = p > 0.0
    accuracy = float(np.mean(binary_truth == binary_pred))

    true_pos = float(np.sum(binary_pred & binary_truth))
    pred_pos = float(np.sum(binary_pred))
    gold_pos = float(np.sum(binary_truth))
    precision = true_pos / pred_pos if pred_pos else 0.0
    recall = true_pos / gold_pos if gold_pos else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return f1, accuracy


def regression_scores(truth: Iterable[float], pred: Iterable[float]) -> Dict[str, float]:
    return {
        "MAE": mae(truth, pred),
        "MSE": mse(truth, pred),
        "Corr": pearson_corr(truth, pred),
    }


def classification_scores(truth: Iterable[float], pred: Iterable[float]) -> Dict[str, float]:
    t, p = _as_1d(truth), _as_1d(pred)
    acc7 = float(np.mean(split_uniform_7(t) == split_uniform_7(p)))
    acc5 = float(np.mean(split_uniform_5(t) == split_uniform_5(p)))
    f1, acc2 = acc2_f1(t, p)
    return {
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def uniform7_edges() -> Tuple[float, ...]:
    step = 6.0 / 7.0
    return tuple(-3.0 + i * step for i in range(8))


def uniform5_edges() -> Tuple[float, ...]:
    step = 6.0 / 5.0
    return tuple(-3.0 + i * step for i in range(6))
