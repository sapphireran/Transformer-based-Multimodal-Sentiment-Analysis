"""Evaluation helpers for multimodal sentiment regression.

The original training loop in ``train_and_test.py`` reports a mix of
regression scores and binned classification scores on the standard
CMU-MOSI / CMU-MOSEI label range ``[-3, 3]``. This module is a
stand-alone, CPU-friendly copy of those calculations so examples and
tests do not have to import the training script (which pulls in
``memory_profiler`` and opens a matplotlib window).

Label conventions
-----------------
* Raw targets are continuous opinion scores, typically in ``[-3, 3]``.
* Binary metrics treat ``score > 0`` as positive and exclude exact zeros
  by default (the usual "non-zero" MOSI/MOSEI protocol).
* Acc-5 / Acc-7 map both predictions and targets onto equal-width bins
  covering ``[-3, 3]``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

ArrayLike = Union[np.ndarray, Iterable[float]]

SENTIMENT_LOW = -3.0
SENTIMENT_HIGH = 3.0
SENTIMENT_SPAN = SENTIMENT_HIGH - SENTIMENT_LOW


def as_1d_numpy(values: ArrayLike) -> np.ndarray:
    """Squeeze a tensor-like object down to a 1-D float array."""
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1)


def eval_affect(truths: ArrayLike, results: ArrayLike, exclude_zero: bool = True) -> tuple[float, float]:
    """Binary F1 and accuracy on the sign of the sentiment score.

    Parameters
    ----------
    truths, results:
        Continuous labels and model outputs.
    exclude_zero:
        When True (default), clips with a ground-truth of exactly 0 are
        dropped, matching the non-zero binary protocol used in the paper
        tables stored under ``model/results/``.

    Returns
    -------
    f1, accuracy
    """
    test_preds = as_1d_numpy(results)
    test_truth = as_1d_numpy(truths)

    if exclude_zero:
        keep = test_truth != 0
    else:
        keep = np.ones(test_truth.shape[0], dtype=bool)

    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0

    if binary_truth.size == 0:
        return 0.0, 0.0

    f1 = float(f1_score(binary_truth, binary_preds, average="binary", zero_division=0))
    accuracy = float(accuracy_score(binary_truth, binary_preds))
    return f1, accuracy


def split_uniform(data: ArrayLike, n_bins: int, low: float = SENTIMENT_LOW, high: float = SENTIMENT_HIGH) -> np.ndarray:
    """Map continuous scores onto ``1 .. n_bins`` equal-width intervals.

    The training script special-cases 5 and 7 bins. This helper is the
    same rule for any positive bin count: edges run from ``low`` to
    ``high`` and ``np.digitize`` assigns 1-based indices, clipped to the
    legal range so a value of exactly ``high`` still lands in the last bin.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    values = as_1d_numpy(data)
    step = (high - low) / n_bins
    edges = [low + i * step for i in range(n_bins + 1)]
    categories = np.digitize(values, edges, right=False)
    return np.clip(categories, 1, n_bins)


def split_uniform_7(data: ArrayLike) -> np.ndarray:
    """Seven equal-width bins over ``[-3, 3]`` (Acc-7)."""
    return split_uniform(data, n_bins=7)


def split_uniform_5(data: ArrayLike) -> np.ndarray:
    """Five equal-width bins over ``[-3, 3]`` (Acc-5)."""
    return split_uniform(data, n_bins=5)


def regression_scores(truths: ArrayLike, preds: ArrayLike) -> Dict[str, float]:
    """MAE, MSE, and Pearson correlation for a clip-level prediction."""
    y_true = as_1d_numpy(truths)
    y_pred = as_1d_numpy(preds)
    if y_true.size == 0:
        return {"MAE": 0.0, "MSE": 0.0, "Corr": 0.0}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    if y_true.size < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        corr = 0.0
    else:
        corr_value, _ = pearsonr(y_true, y_pred)
        corr = float(0.0 if np.isnan(corr_value) else corr_value)
    return {"MAE": mae, "MSE": mse, "Corr": corr}


def classification_scores(truths: ArrayLike, preds: ArrayLike, exclude_zero: bool = True) -> Dict[str, float]:
    """Acc-7, Acc-5, binary Acc-2, and binary F1."""
    y_true = as_1d_numpy(truths)
    y_pred = as_1d_numpy(preds)
    acc7 = float(accuracy_score(split_uniform_7(y_true), split_uniform_7(y_pred)))
    acc5 = float(accuracy_score(split_uniform_5(y_true), split_uniform_5(y_pred)))
    f1, acc2 = eval_affect(y_true, y_pred, exclude_zero=exclude_zero)
    return {
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def evaluate_sentiment(truths: ArrayLike, preds: ArrayLike, exclude_zero: bool = True) -> Dict[str, float]:
    """Full metric bundle used in the published result CSVs.

    Keys match ``single_test`` in ``train_and_test.py`` so a toy example
    and a full MOSEI run can be compared side by side.
    """
    scores = {}
    scores.update(regression_scores(truths, preds))
    scores.update(classification_scores(truths, preds, exclude_zero=exclude_zero))
    return scores


def format_score_table(rows: Mapping[str, Mapping[str, float]], digits: int = 4) -> str:
    """Pretty-print a ``{method: {metric: value}}`` mapping as Markdown."""
    if not rows:
        return "_no scores_"
    metrics = list(next(iter(rows.values())).keys())
    header = "| Method | " + " | ".join(metrics) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(metrics)) + " |"
    lines = [header, sep]
    for name, scores in rows.items():
        cells = [name] + [f"{scores[m]:.{digits}f}" if isinstance(scores.get(m), (int, float)) else "-" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
