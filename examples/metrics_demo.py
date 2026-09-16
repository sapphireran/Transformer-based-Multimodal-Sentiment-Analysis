#!/usr/bin/env python3
"""Show how a scalar in [-3, 3] becomes Acc-7 / Acc-5 / Acc-2 / F1.

Uses the same helpers as ``model/train_and_test.py`` so a protocol change
there shows up here and in ``tests/test_metrics.py``.
"""

from __future__ import annotations

import numpy as np
import torch

import common  # noqa: F401  — puts model/ on sys.path
from train_and_test import eval_affect, split_uniform_5, split_uniform_7


def _bin_table(values: np.ndarray) -> None:
    acc7 = split_uniform_7(values)
    acc5 = split_uniform_5(values)
    print(f"{'score':>8}  {'Acc-7':>6}  {'Acc-5':>6}")
    for score, a7, a5 in zip(values, acc7, acc5):
        print(f"{score:8.3f}  {int(a7):6d}  {int(a5):6d}")


def main() -> None:
    print("Uniform bin edges on [-3, 3]")
    print("  Acc-7 width = 6/7 ≈ 0.8571   Acc-5 width = 1.2")
    print("  digitize is left-closed / right-open except the last bin, then clip to 1..K\n")

    probe = np.array([-3.0, -2.2, -1.0, -0.1, 0.0, 0.1, 1.0, 2.2, 3.0])
    _bin_table(probe)

    print("\nPolarity (Acc-2 / F1), exclude_zero=True — neutrals dropped")
    truth = torch.tensor([[-2.0], [-0.4], [0.0], [0.3], [1.8]])
    pred = torch.tensor([[-1.1], [0.2], [0.0], [0.9], [1.2]])
    f1, acc2 = eval_affect(truth, pred, exclude_zero=True)
    kept = [i for i, y in enumerate(truth.view(-1).tolist()) if y != 0.0]
    print(f"  y     = {truth.view(-1).tolist()}")
    print(f"  yhat  = {pred.view(-1).tolist()}")
    print(f"  kept indices (y != 0): {kept}")
    print(f"  Acc-2 = {acc2:.4f}   F1 = {f1:.4f}")

    print("\nSame vectors with exclude_zero=False (0 counts as negative, because > 0 is the cutoff)")
    f1_all, acc2_all = eval_affect(truth, pred, exclude_zero=False)
    print(f"  Acc-2 = {acc2_all:.4f}   F1 = {f1_all:.4f}")

    print("\nRegression extras the eval script also prints")
    y = truth.view(-1)
    yhat = pred.view(-1)
    mae = torch.mean(torch.abs(y - yhat)).item()
    mse = torch.mean((y - yhat) ** 2).item()
    # Pearson on the five-point toy vectors
    y_c = y - y.mean()
    yhat_c = yhat - yhat.mean()
    corr = (y_c * yhat_c).sum() / (y_c.norm() * yhat_c.norm())
    print(f"  MAE = {mae:.4f}   MSE = {mse:.4f}   Corr = {corr.item():.4f}")
    print(
        "\nCSV columns Acc7_uniform / Acc5_uniform use the digitize bins above,"
        " not integer rounding to {-3,...,3}."
    )


if __name__ == "__main__":
    main()
