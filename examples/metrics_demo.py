"""Headless copy of the metric helpers in model/train_and_test.py.

single_test() also draws a confusion matrix with plt.show(), which blocks
in this environment. The functions below are the same arithmetic so you
can see what a CSV row means on a 6-utterance toy set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr


def split_uniform_7(data: np.ndarray) -> np.ndarray:
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data: np.ndarray) -> np.ndarray:
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 5)


def eval_affect(truths: np.ndarray, results: np.ndarray, exclude_zero: bool = True):
    truths = np.asarray(truths).reshape(-1)
    results = np.asarray(results).reshape(-1)
    keep = np.array([i for i, e in enumerate(truths) if e != 0 or (not exclude_zero)])
    binary_truth = truths[keep] > 0
    binary_preds = results[keep] > 0
    return (
        float(f1_score(binary_truth, binary_preds, average="binary")),
        float(accuracy_score(binary_truth, binary_preds)),
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    corr, _ = pearsonr(y_true, y_pred)
    acc7 = float(accuracy_score(split_uniform_7(y_true), split_uniform_7(y_pred)))
    acc5 = float(accuracy_score(split_uniform_5(y_true), split_uniform_5(y_pred)))
    f1, acc2 = eval_affect(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "Corr": float(corr),
        "Acc7_uniform": acc7,
        "Acc5_uniform": acc5,
        "Acc2": acc2,
        "F1": f1,
    }


def _bin_table() -> str:
    step7 = 6.0 / 7.0
    edges7 = [-3.0 + i * step7 for i in range(8)]
    rows = ["Acc7 bin edges (uniform [-3, 3]):"]
    for i in range(7):
        rows.append(f"  bin {i + 1}: [{edges7[i]: .4f}, {edges7[i + 1]: .4f})")
    return "\n".join(rows)


def main() -> None:
    # Hand-chosen pairs: two neutrals, two positives, two negatives.
    y_true = np.array([-2.4, -0.8, 0.0, 0.0, 1.1, 2.6])
    y_pred = np.array([-2.1, -0.2, 0.1, -0.3, 0.9, 2.0])

    print("y_true:", y_true.tolist())
    print("y_pred:", y_pred.tolist())
    print()
    print(_bin_table())
    print()
    print("uniform-7 codes  true", split_uniform_7(y_true).tolist())
    print("uniform-7 codes  pred", split_uniform_7(y_pred).tolist())
    print("uniform-5 codes  true", split_uniform_5(y_true).tolist())
    print("uniform-5 codes  pred", split_uniform_5(y_pred).tolist())
    print()

    metrics = evaluate(y_true, y_pred)
    print("CSV-style row:")
    for key in ("MAE", "Acc7_uniform", "Acc5_uniform", "Acc2", "Corr", "F1"):
        print(f"  {key:14s} {metrics[key]:.4f}")
    print(f"  {'MSE':14s} {metrics['MSE']:.4f}")
    print()
    print("Acc2 / F1 drop the two gold-zero utterances, then test (score > 0).")
    print("That leaves four items: two negative gold, two positive gold.")


if __name__ == "__main__":
    main()
