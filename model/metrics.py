"""Sentiment evaluation helpers used by the examples lab and unit tests.

The training scripts in ``train_and_test.py`` compute the same family of
scores after a full MOSI / MOSEI pass. This module is the dataset-free
version: it takes numpy arrays (or tensors) of continuous scores in
``[-3, 3]`` and returns the regression and classification numbers that
appear in the result CSVs.

Uniform Acc-5 / Acc-7
---------------------
CMU-MOSEI papers often bin the Likert-style score with *fixed* cut points
such as ``{-2, -1, 0, 1, 2}``. This repository instead splits ``[-3, 3]``
into equal-width bins (see ``split_uniform_7`` / ``split_uniform_5``). The
published tables in ``model/results/`` use that uniform mapping, so the
examples keep it.

Binary Acc / F1
---------------
``eval_affect`` follows the common MOSI protocol: drop clips whose true
score is exactly 0 (neutral) unless ``exclude_zero=False``, then treat
``score > 0`` as positive.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

SENTIMENT_LOW = -3.0
SENTIMENT_HIGH = 3.0
SENTIMENT_SPAN = SENTIMENT_HIGH - SENTIMENT_LOW


def as_1d_numpy(values) -> np.ndarray:
    """Convert a tensor / list / ndarray of scores to a 1-D float array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("score array is empty")
    return array


def split_uniform(data, n_bins: int) -> np.ndarray:
    """Map scores in ``[-3, 3]`` onto integer labels ``1 .. n_bins``.

    Matches ``split_uniform_7`` / ``split_uniform_5`` in ``train_and_test.py``:
    ``numpy.digitize`` with left-closed bins, then clip overflow from the
    right endpoint ``+3.0``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    data = as_1d_numpy(data)
    step = SENTIMENT_SPAN / float(n_bins)
    edges = [SENTIMENT_LOW + i * step for i in range(n_bins + 1)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, n_bins)


def split_uniform_7(data) -> np.ndarray:
    """Seven equal-width bins over ``[-3, 3]`` (labels 1..7)."""
    return split_uniform(data, 7)


def split_uniform_5(data) -> np.ndarray:
    """Five equal-width bins over ``[-3, 3]`` (labels 1..5)."""
    return split_uniform(data, 5)


def eval_affect(truths, results, exclude_zero: bool = True) -> tuple[float, float]:
    """Binary F1 and accuracy on the positive/negative split.

    Parameters
    ----------
    truths, results:
        Continuous sentiment scores. Neutral truths (exactly 0) are
        dropped when ``exclude_zero`` is true, matching MOSI practice.
    """
    test_truth = as_1d_numpy(truths)
    test_preds = as_1d_numpy(results)
    if test_truth.shape != test_preds.shape:
        raise ValueError(
            f"truth/pred length mismatch: {test_truth.shape} vs {test_preds.shape}"
        )

    keep = np.array(
        [i for i, value in enumerate(test_truth) if value != 0 or (not exclude_zero)],
        dtype=np.int64,
    )
    if keep.size == 0:
        return float("nan"), float("nan")

    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0
    f1 = float(f1_score(binary_truth, binary_preds, average="binary"))
    accuracy = float(accuracy_score(binary_truth, binary_preds))
    return f1, accuracy


def regression_scores(truths, results) -> dict[str, float]:
    """MAE, MSE, and Pearson correlation for continuous sentiment."""
    y_true = as_1d_numpy(truths)
    y_pred = as_1d_numpy(results)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"truth/pred length mismatch: {y_true.shape} vs {y_pred.shape}")
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if y_true.size < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        corr = float("nan")
    else:
        corr = float(pearsonr(y_true, y_pred)[0])
    return {"MSE": mse, "MAE": mae, "Corr": corr}


def classification_scores(truths, results, exclude_zero: bool = True) -> dict[str, float]:
    """Uniform Acc-7 / Acc-5 plus binary Acc-2 / F1."""
    y_true = as_1d_numpy(truths)
    y_pred = as_1d_numpy(results)
    acc7 = float(accuracy_score(split_uniform_7(y_true), split_uniform_7(y_pred)))
    acc5 = float(accuracy_score(split_uniform_5(y_true), split_uniform_5(y_pred)))
    f1, acc2 = eval_affect(y_true, y_pred, exclude_zero=exclude_zero)
    return {
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def evaluate_sentiment(truths, results, exclude_zero: bool = True) -> dict[str, float]:
    """Full metric dict used throughout this personal study."""
    scores: dict[str, float] = {}
    scores.update(regression_scores(truths, results))
    scores.update(classification_scores(truths, results, exclude_zero=exclude_zero))
    return scores


def round_result_row(scores: Mapping[str, float], digits: int = 4) -> dict[str, float]:
    """Round a metric dict the same way the training scripts write CSVs."""
    rounded: dict[str, float] = {}
    for key, value in scores.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            rounded[key] = float("nan")
        else:
            rounded[key] = round(float(value), digits)
    return rounded


CSV_COLUMNS: Sequence[str] = (
    "Fusion Method",
    "MAE",
    "ACC7",
    "Acc5",
    "ACC2",
    "Corr",
    "F1",
)


def row_for_csv(name: str, scores: Mapping[str, float]) -> list:
    """Build one CSV row matching ``model/results/*.csv`` headers."""
    rounded = round_result_row(scores)
    return [
        name,
        rounded.get("MAE"),
        rounded.get("Acc7_uniform"),
        rounded.get("Acc5_uniform"),
        rounded.get("Acc2"),
        rounded.get("Corr"),
        rounded.get("F1"),
    ]


def format_score_table(named_scores: Sequence[tuple[str, Mapping[str, float]]]) -> str:
    """Pretty-print a comparison table for the examples lab."""
    headers = ["name", "MAE", "Acc7", "Acc5", "Acc2", "Corr", "F1"]
    rows = [headers]
    for name, scores in named_scores:
        rounded = round_result_row(scores)
        rows.append(
            [
                name,
                f"{rounded.get('MAE', float('nan')):.4f}",
                f"{rounded.get('Acc7_uniform', float('nan')):.4f}",
                f"{rounded.get('Acc5_uniform', float('nan')):.4f}",
                f"{rounded.get('Acc2', float('nan')):.4f}",
                f"{rounded.get('Corr', float('nan')):.4f}",
                f"{rounded.get('F1', float('nan')):.4f}",
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for index, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(lines)


def summarize_bin_edges(n_bins: int) -> list[tuple[int, float, float]]:
    """Return ``(label, left, right)`` for the uniform sentiment bins."""
    step = SENTIMENT_SPAN / float(n_bins)
    edges = [SENTIMENT_LOW + i * step for i in range(n_bins + 1)]
    return [
        (i + 1, edges[i], edges[i + 1])
        for i in range(n_bins)
    ]
