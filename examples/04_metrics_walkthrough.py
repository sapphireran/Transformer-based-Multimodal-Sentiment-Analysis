#!/usr/bin/env python3
"""Worked examples of the affect metrics used in ``single_test``.

No model is loaded. The numbers are small enough to check by hand so the
bin edges and the zero-exclusion rule are obvious.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import bin_edges, eval_affect, regression_and_bins, split_uniform_5, split_uniform_7


def show_edges() -> None:
    print("=== Uniform bin edges on [-3, 3] ===")
    for n in (5, 7):
        edges = bin_edges(n)
        print(f"Acc-{n} edges ({n} bins, {n + 1} numbers):")
        print("  " + "  ".join(f"{e:+.3f}" for e in edges))
        print(f"  width = {6.0 / n:.4f}")
    print()


def show_digitize() -> None:
    print("=== Digitizing a few gold labels (Acc-7) ===")
    values = np.array([-3.0, -2.1, -0.01, 0.0, 0.4, 2.9, 3.0])
    cats = split_uniform_7(values)
    edges = bin_edges(7)
    for value, cat in zip(values, cats):
        lo = edges[cat - 1]
        hi = edges[cat]
        print(f"  {value:+5.2f}  → bin {cat}  [{lo:+.3f}, {hi:+.3f})")
    print("  note: 3.0 is clipped into the last bin even though digitize can overflow.")
    print()


def show_perfect_and_sign_flip() -> None:
    print("=== Regression + bin metrics on two toy vectors ===")
    truth = np.array([-2.0, -0.5, 0.0, 0.8, 2.4])
    perfect = truth.copy()
    flipped = -truth
    constant = np.zeros_like(truth)

    for name, pred in ("perfect", perfect), ("sign-flipped", flipped), ("all-zero", constant):
        scores = regression_and_bins(truth, pred)
        print(f"  {name:13s}  MAE={scores['MAE']:.3f}  Corr={scores['Corr']:+.3f}  "
              f"Acc7={scores['Acc7_uniform']:.3f}  Acc2={scores['Acc2']:.3f}  F1={scores['F1']:.3f}")
    print()
    print("Acc-2 / F1 drop the gold-zero clip, so the middle 0.0 row is ignored there.")
    f1, acc2 = eval_affect(truth, flipped, exclude_zero=True)
    f1_keep, acc2_keep = eval_affect(truth, flipped, exclude_zero=False)
    print(f"  sign-flipped Acc-2 exclude_zero=True  → Acc2={acc2:.3f} F1={f1:.3f}")
    print(f"  sign-flipped Acc-2 exclude_zero=False → Acc2={acc2_keep:.3f} F1={f1_keep:.3f}")
    print()


def show_acc5_vs_acc7() -> None:
    print("=== Same pair of scores, Acc-5 vs Acc-7 ===")
    truth = np.array([0.2, 0.9])
    pred = np.array([0.3, 1.6])
    print(f"  gold {truth.tolist()}  pred {pred.tolist()}")
    print(f"  Acc-7 bins gold={split_uniform_7(truth).tolist()} pred={split_uniform_7(pred).tolist()}")
    print(f"  Acc-5 bins gold={split_uniform_5(truth).tolist()} pred={split_uniform_5(pred).tolist()}")
    scores = regression_and_bins(truth, pred)
    print(f"  Acc7={scores['Acc7_uniform']:.2f}  Acc5={scores['Acc5_uniform']:.2f}  MAE={scores['MAE']:.3f}")
    print()


def main() -> None:
    show_edges()
    show_digitize()
    show_perfect_and_sign_flip()
    show_acc5_vs_acc7()
    print("These are the same definitions ``model/train_and_test.py`` writes into the CSVs.")


if __name__ == "__main__":
    main()
