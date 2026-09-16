"""Sentiment evaluation helpers shared by training scripts and examples.

CMU-MOSI / CMU-MOSEI labels are continuous in ``[-3, 3]``. The training loop
regresses that score with L1/MSE, then this module derives:

* regression quality (MAE, MSE, Pearson correlation)
* 7-class and 5-class accuracy on uniform bins of ``[-3, 3]``
* binary accuracy / F1 after dropping (optional) exact-zero labels

The bin edges match ``train_and_test.split_uniform_*`` so example scripts and
the original evaluation path report the same numbers.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # type: ignore[name-defined]


def as_numpy(values: ArrayLike) -> np.ndarray:
    """Convert a tensor or ndarray to a 1-d numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1)


def eval_affect(truths: ArrayLike, results: ArrayLike, exclude_zero: bool = True):
    """Binary F1 and accuracy on the sign of the sentiment score.

    Samples whose label is exactly 0 are dropped when ``exclude_zero`` is true,
    matching the Multimodal Transformer / CMU-MultimodalSDK convention.
    """
    test_preds = as_numpy(results)
    test_truth = as_numpy(truths)

    if exclude_zero:
        keep = test_truth != 0
        test_preds = test_preds[keep]
        test_truth = test_truth[keep]

    binary_truth = test_truth > 0
    binary_preds = test_preds > 0
    f1 = f1_score(binary_truth, binary_preds, average="binary", zero_division=0)
    accuracy = accuracy_score(binary_truth, binary_preds)
    return f1, accuracy


def uniform_bin_edges(n_bins: int, low: float = -3.0, high: float = 3.0) -> np.ndarray:
    """Return the ``n_bins + 1`` edges used by ``split_uniform``.

    ``np.digitize(..., right=False)`` assigns ``x`` to bin ``i`` when
    ``edges[i-1] <= x < edges[i]``, then the result is clipped to
    ``1 .. n_bins``. The last bin therefore also absorbs ``x == high``
    and anything larger.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    step = (high - low) / n_bins
    return np.asarray([low + i * step for i in range(n_bins + 1)], dtype=np.float64)


def bin_intervals(n_bins: int, low: float = -3.0, high: float = 3.0):
    """List ``(index, left, right)`` triples for the uniform bins."""
    edges = uniform_bin_edges(n_bins, low=low, high=high)
    intervals = []
    for i in range(n_bins):
        intervals.append((i + 1, float(edges[i]), float(edges[i + 1])))
    return intervals


def split_uniform(data: ArrayLike, n_bins: int, low: float = -3.0, high: float = 3.0) -> np.ndarray:
    """Map scores in ``[low, high]`` onto ``1 .. n_bins`` using equal-width bins."""
    values = as_numpy(data)
    edges = uniform_bin_edges(n_bins, low=low, high=high)
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, n_bins)


def split_uniform_7(data: ArrayLike) -> np.ndarray:
    """Split ``[-3, 3]`` into 7 equal bins (Acc-7)."""
    return split_uniform(data, n_bins=7)


def split_uniform_5(data: ArrayLike) -> np.ndarray:
    """Split ``[-3, 3]`` into 5 equal bins (Acc-5)."""
    return split_uniform(data, n_bins=5)


def compute_sentiment_metrics(
    truths: ArrayLike,
    predictions: ArrayLike,
    exclude_zero: bool = True,
) -> MutableMapping[str, float]:
    """Return the metric dict used by ``train_and_test.single_test``.

    Keys are kept identical to the training scripts so CSV writers and example
    printers can share a schema:

    ``MAE``, ``MSE``, ``Corr``, ``Acc7_uniform``, ``Acc5_uniform``, ``Acc2``, ``F1``.
    """
    true_vals = as_numpy(truths)
    pred_vals = as_numpy(predictions)
    if true_vals.shape != pred_vals.shape:
        raise ValueError(
            f"truth/prediction shape mismatch: {true_vals.shape} vs {pred_vals.shape}"
        )

    if true_vals.size == 0:
        raise ValueError("cannot score an empty prediction array")

    if true_vals.size == 1 or np.allclose(true_vals, true_vals[0]) or np.allclose(pred_vals, pred_vals[0]):
        corr = 0.0
    else:
        corr_value, _ = pearsonr(true_vals, pred_vals)
        corr = 0.0 if np.isnan(corr_value) else float(corr_value)

    mse = float(np.mean((true_vals - pred_vals) ** 2))
    mae = float(np.mean(np.abs(true_vals - pred_vals)))

    pred_7 = split_uniform_7(pred_vals)
    true_7 = split_uniform_7(true_vals)
    acc7 = float(accuracy_score(true_7, pred_7))

    pred_5 = split_uniform_5(pred_vals)
    true_5 = split_uniform_5(true_vals)
    acc5 = float(accuracy_score(true_5, pred_5))

    f1, acc2 = eval_affect(true_vals, pred_vals, exclude_zero=exclude_zero)

    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": corr,
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": float(acc2),
        "F1": float(f1),
    }


def format_metrics(metrics: Mapping[str, float], precision: int = 4) -> str:
    """Pretty-print a metric dict as aligned ``name: value`` lines."""
    width = max(len(str(key)) for key in metrics)
    lines = []
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            lines.append(f"{str(key):<{width}}  {float(value):.{precision}f}")
        else:
            lines.append(f"{str(key):<{width}}  {value}")
    return "\n".join(lines)


__all__ = [
    "as_numpy",
    "bin_intervals",
    "compute_sentiment_metrics",
    "eval_affect",
    "format_metrics",
    "split_uniform",
    "split_uniform_5",
    "split_uniform_7",
    "uniform_bin_edges",
]
