"""Headless copy of the evaluation arithmetic in ``train_and_test``.

``single_test`` opens a matplotlib window and needs a DataLoader.
This module is the same math on two 1-d tensors so the examples (and
any later notebook) can score a batch without those dependencies.

Binning rules match ``split_uniform_7`` / ``split_uniform_5``:

* the gold range is assumed to be ``[-3, 3]``
* Acc-7 / Acc-5 use equal-width bins, not "round to nearest integer"
* Acc-2 / F1 drop exact-zero gold labels by default
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _as_1d(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("expected a non-empty vector")
    return values


def split_uniform(data, n_bins: int) -> np.ndarray:
    """Map each score in ``[-3, 3]`` to a 1-based bin index.

    ``n_bins=7`` and ``n_bins=5`` reproduce the helpers in
    ``model/train_and_test.py``. Values outside the interval are
    clipped into the first or last bin so Acc-7 never sees an 8.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    values = _as_1d(data)
    step = 6.0 / n_bins
    edges = [-3.0 + i * step for i in range(n_bins + 1)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, n_bins)


def split_uniform_7(data) -> np.ndarray:
    return split_uniform(data, 7)


def split_uniform_5(data) -> np.ndarray:
    return split_uniform(data, 5)


def eval_affect(truths, results, exclude_zero: bool = True) -> tuple[float, float]:
    """Binary F1 and accuracy on ``sign(score)``, optionally dropping zeros.

    Returns ``(f1, accuracy)`` in that order, matching
    ``train_and_test.eval_affect``.
    """
    y = _as_1d(truths)
    y_hat = _as_1d(results)
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: gold {y.shape} vs pred {y_hat.shape}")
    keep = np.ones(y.shape[0], dtype=bool) if not exclude_zero else (y != 0)
    if not np.any(keep):
        raise ValueError("no labels left after dropping zeros")
    binary_truth = y[keep] > 0
    binary_pred = y_hat[keep] > 0
    return (
        float(f1_score(binary_truth, binary_pred, average="binary")),
        float(accuracy_score(binary_truth, binary_pred)),
    )


def pearson_corr(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Pearson r, or ``nan`` when either vector is constant."""
    if np.std(y) == 0.0 or np.std(y_hat) == 0.0:
        return float("nan")
    return float(np.corrcoef(y, y_hat)[0, 1])


@dataclass
class MetricReport:
    mae: float
    mse: float
    corr: float
    acc7: float
    acc5: float
    acc2: float
    f1: float
    pred_7: np.ndarray = field(repr=False)
    true_7: np.ndarray = field(repr=False)
    n: int = 0
    n_nonzero: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "MAE": self.mae,
            "MSE": self.mse,
            "Corr": self.corr,
            "Acc7_uniform": self.acc7,
            "Acc5_uniform": self.acc5,
            "Acc2": self.acc2,
            "F1": self.f1,
        }

    def pretty(self) -> str:
        rows = [
            f"n={self.n}  n_nonzero={self.n_nonzero}",
            f"MAE  {self.mae:.4f}   MSE {self.mse:.4f}   Corr {self.corr:.4f}",
            f"Acc7 {self.acc7:.4f}   Acc5 {self.acc5:.4f}",
            f"Acc2 {self.acc2:.4f}   F1   {self.f1:.4f}",
        ]
        return "\n".join(rows)


def evaluate_regression(truths, results, exclude_zero: bool = True) -> MetricReport:
    """Score a pair of gold / prediction vectors with the repo protocol."""
    y = _as_1d(truths)
    y_hat = _as_1d(results)
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: gold {y.shape} vs pred {y_hat.shape}")

    mae = float(np.mean(np.abs(y_hat - y)))
    mse = float(np.mean((y_hat - y) ** 2))
    corr = pearson_corr(y, y_hat)

    pred_7 = split_uniform_7(y_hat)
    true_7 = split_uniform_7(y)
    pred_5 = split_uniform_5(y_hat)
    true_5 = split_uniform_5(y)
    acc7 = float(accuracy_score(true_7, pred_7))
    acc5 = float(accuracy_score(true_5, pred_5))
    f1, acc2 = eval_affect(y, y_hat, exclude_zero=exclude_zero)

    return MetricReport(
        mae=mae,
        mse=mse,
        corr=corr,
        acc7=acc7,
        acc5=acc5,
        acc2=acc2,
        f1=f1,
        pred_7=pred_7,
        true_7=true_7,
        n=int(y.size),
        n_nonzero=int(np.count_nonzero(y)),
    )
