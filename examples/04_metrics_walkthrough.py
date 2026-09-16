#!/usr/bin/env python3
"""Show how Acc7 / Acc5 / Acc2 / F1 are derived from a scalar score.

    python examples/04_metrics_walkthrough.py
"""

from __future__ import annotations

import numpy as np

import common  # noqa: F401  — puts model/ on sys.path
from metrics import (
    compute_affect_metrics,
    eval_affect,
    format_metrics_row,
    split_round_7,
    split_uniform_5,
    split_uniform_7,
)


def show_bins() -> None:
    probes = np.array([-3.0, -2.5, -1.5, -0.1, 0.0, 0.1, 1.5, 2.5, 3.0])
    print("Uniform bin edges on [-3, 3]")
    print("  Acc7 step = 6/7 = {:.4f}".format(6.0 / 7.0))
    print("  Acc5 step = 6/5 = {:.4f}".format(6.0 / 5.0))
    print()
    print(f"{'y':>6}  {'uniform7':>8}  {'round7':>6}  {'uniform5':>8}")
    for value, u7, r7, u5 in zip(
        probes, split_uniform_7(probes), split_round_7(probes), split_uniform_5(probes)
    ):
        print(f"{value:6.1f}  {int(u7):8d}  {int(r7):6d}  {int(u5):8d}")
    print()
    print(
        "Published tables in this repo use uniform7 / uniform5, not the rounded Acc7."
    )


def show_exclude_zero() -> None:
    truth = np.array([-2.0, -0.4, 0.0, 0.0, 0.8, 2.2])
    pred = np.array([-1.0, 0.3, 0.1, -0.2, 1.1, 1.8])
    f1_ex, acc_ex = eval_affect(truth, pred, exclude_zero=True)
    f1_all, acc_all = eval_affect(truth, pred, exclude_zero=False)
    print("Acc2 / F1 with and without dropping y == 0")
    print(f"  gold: {truth.tolist()}")
    print(f"  pred: {pred.tolist()}")
    print(f"  exclude_zero=True  Acc2={acc_ex:.3f}  F1={f1_ex:.3f}   (4 non-neutral)")
    print(f"  exclude_zero=False Acc2={acc_all:.3f}  F1={f1_all:.3f}  (6 points)")
    print()


def show_perfect_and_sign_flip() -> None:
    truth = np.array([-2.4, -1.1, 0.6, 1.8, 2.7])
    perfect = compute_affect_metrics(truth, truth)
    flipped = compute_affect_metrics(truth, -truth)
    shifted = compute_affect_metrics(truth, truth + 0.4)
    print("| case | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(format_metrics_row("perfect copy", perfect))
    print(format_metrics_row("sign flip", flipped))
    print(format_metrics_row("plus 0.4", shifted))


def main() -> int:
    show_bins()
    print()
    show_exclude_zero()
    show_perfect_and_sign_flip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
