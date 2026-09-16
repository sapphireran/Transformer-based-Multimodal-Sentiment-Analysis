"""Walk the evaluation helpers on a tiny hand-built (y, ŷ) batch.

This is the same math ``single_test`` uses, without loading a model or
opening a matplotlib window.

    python examples/metrics_demo.py
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from common import MODEL_DIR  # noqa: F401  (puts model/ on sys.path)

from train_and_test import eval_affect, split_uniform_5, split_uniform_7

# Five clips: two positive, two negative, one gold-neutral.
Y = np.array([2.0, -1.0, 0.5, -2.5, 0.0], dtype=np.float64)
YHAT = np.array([1.8, -0.8, 0.2, -2.0, 0.1], dtype=np.float64)


def metrics_from_repo(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(yhat - y)))
    mse = float(np.mean((yhat - y) ** 2))
    corr, _ = pearsonr(y, yhat)
    acc7 = float(accuracy_score(split_uniform_7(y), split_uniform_7(yhat)))
    acc5 = float(accuracy_score(split_uniform_5(y), split_uniform_5(yhat)))
    f1, acc2 = eval_affect(y, yhat, exclude_zero=True)
    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": float(corr),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": float(acc2),
        "F1": float(f1),
    }


def metrics_by_hand(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """Independent copy used as a test oracle."""

    mae = float(np.mean(np.abs(yhat - y)))
    mse = float(np.mean((yhat - y) ** 2))
    corr, _ = pearsonr(y, yhat)

    def bins(data, k):
        step = 6.0 / k
        edges = [-3.0 + i * step for i in range(k + 1)]
        return np.clip(np.digitize(data, edges, right=False), 1, k)

    acc7 = float(accuracy_score(bins(y, 7), bins(yhat, 7)))
    acc5 = float(accuracy_score(bins(y, 5), bins(yhat, 5)))

    keep = y != 0
    binary_truth = y[keep] > 0
    binary_pred = yhat[keep] > 0
    acc2 = float(accuracy_score(binary_truth, binary_pred))
    f1 = float(f1_score(binary_truth, binary_pred, average="binary"))
    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": float(corr),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def run() -> Dict[str, float]:
    repo = metrics_from_repo(Y, YHAT)
    hand = metrics_by_hand(Y, YHAT)
    print("Hand-built batch")
    print(f"  y    = {Y.tolist()}")
    print(f"  yhat = {YHAT.tolist()}")
    print()
    print(f"{'metric':<16} {'repo':>10} {'hand':>10}")
    for key in repo:
        print(f"{key:<16} {repo[key]:10.4f} {hand[key]:10.4f}")
        if abs(repo[key] - hand[key]) > 1e-9:
            raise AssertionError(f"{key}: repo {repo[key]} != hand {hand[key]}")

    # Gold zeros are dropped: four signed clips, all signs match → Acc-2 = 1.
    if abs(repo["Acc2"] - 1.0) > 1e-12:
        raise AssertionError("expected Acc-2 = 1.0 on this toy batch")
    if abs(repo["MAE"] - 0.26) > 1e-12:
        raise AssertionError(f"expected MAE = 0.26, got {repo['MAE']}")

    print()
    print("split_uniform_7(y)    ", split_uniform_7(Y).tolist())
    print("split_uniform_7(yhat) ", split_uniform_7(YHAT).tolist())
    print("split_uniform_5(y)    ", split_uniform_5(Y).tolist())
    print("split_uniform_5(yhat) ", split_uniform_5(YHAT).tolist())
    return repo


if __name__ == "__main__":
    run()
