"""Recompute the repo's evaluation numbers on a tiny hand-checkable set.

This is a standalone port of ``eval_affect`` / ``split_uniform_7`` /
``split_uniform_5`` plus MAE / MSE / Pearson r, so the arithmetic is
visible without importing ``train_and_test`` (which pulls
``memory_profiler`` and opens a confusion-matrix window).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(truth - pred)))


def mse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((truth - pred) ** 2))


def pearson_r(truth: np.ndarray, pred: np.ndarray) -> float:
    t = np.asarray(truth, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = t - t.mean()
    p = p - p.mean()
    denom = np.sqrt((t * t).sum() * (p * p).sum())
    if denom == 0:
        return float("nan")
    return float((t * p).sum() / denom)


def split_uniform(data: Iterable[float], n_bins: int) -> np.ndarray:
    """Equal-width bins on [-3, 3], matching train_and_test.py."""
    data = np.asarray(list(data), dtype=np.float64).reshape(-1)
    step = 6.0 / n_bins
    edges = [-3.0 + i * step for i in range(n_bins + 1)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, n_bins)


def eval_affect(truths, results, exclude_zero: bool = True):
    """Binary F1 + Acc-2 after optionally dropping neutrals."""
    test_preds = np.asarray(results, dtype=np.float64).reshape(-1)
    test_truth = np.asarray(truths, dtype=np.float64).reshape(-1)
    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    f1 = f1_score(binary_truth, binary_preds, average="binary")
    accuracy = accuracy_score(binary_truth, binary_preds)
    return float(f1), float(accuracy)


def score_regression(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    f1, acc2 = eval_affect(truth, pred)
    acc7 = float(accuracy_score(split_uniform(truth, 7), split_uniform(pred, 7)))
    acc5 = float(accuracy_score(split_uniform(truth, 5), split_uniform(pred, 5)))
    return {
        "MAE": mae(truth, pred),
        "MSE": mse(truth, pred),
        "Corr": pearson_r(truth, pred),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


# Twelve points chosen so each metric can be checked with a calculator.
DEMO_TRUTH = np.array(
    [-2.8, -2.0, -1.1, -0.2, 0.0, 0.0, 0.4, 0.9, 1.4, 2.1, 2.7, 3.0]
)
DEMO_PRED = np.array(
    [-2.4, -1.2, -1.3, 0.3, 0.1, -0.4, 0.2, 1.1, 1.0, 1.6, 2.9, 2.4]
)


def _print_bin_edges(n_bins: int) -> None:
    step = 6.0 / n_bins
    edges = [-3.0 + i * step for i in range(n_bins + 1)]
    pretty = ", ".join(f"{e:.3f}" for e in edges)
    print(f"  Acc-{n_bins} edges: [{pretty}]")


def _demo() -> dict[str, float]:
    print("metrics walkthrough (12 toy clips)")
    print("  truth:", np.array2string(DEMO_TRUTH, precision=1))
    print("  pred: ", np.array2string(DEMO_PRED, precision=1))
    _print_bin_edges(7)
    _print_bin_edges(5)
    print("  Acc-7 bins truth:", split_uniform(DEMO_TRUTH, 7))
    print("  Acc-7 bins pred: ", split_uniform(DEMO_PRED, 7))
    scores = score_regression(DEMO_TRUTH, DEMO_PRED)
    expected_mae = float(np.mean(np.abs(DEMO_TRUTH - DEMO_PRED)))
    if abs(scores["MAE"] - expected_mae) > 1e-12:
        raise AssertionError(f"MAE {scores['MAE']} != {expected_mae}")
    for key in ("MAE", "MSE", "Corr", "Acc7_uniform", "Acc5_uniform", "Acc2", "F1"):
        print(f"  {key:14s} {scores[key]:.4f}")
    # Neutrals at indices 4 and 5 must be ignored for Acc-2.
    f1_keep, acc_keep = eval_affect(DEMO_TRUTH, DEMO_PRED, exclude_zero=False)
    print(f"  Acc2 if neutrals kept: {acc_keep:.4f} (F1 {f1_keep:.4f})")
    return scores


if __name__ == "__main__":
    _demo()
