#!/usr/bin/env python3
"""Hand-checked walkthrough of the personal evaluation metrics.

Reimplements the Acc7 / Acc5 edges and Acc2 / F1 rules from
model/train_and_test.py without importing that file (it pulls in
memory_profiler and opens a matplotlib window).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.metrics import (
    acc2_f1,
    classification_scores,
    regression_scores,
    split_uniform_5,
    split_uniform_7,
    uniform5_edges,
    uniform7_edges,
)


def _print_edges() -> None:
    print("Uniform Acc7 edges (width 6/7):")
    edges = uniform7_edges()
    for i, (lo, hi) in enumerate(zip(edges, edges[1:]), start=1):
        print(f"  class {i}: [{lo:.6f}, {hi:.6f})")
    print()
    print("Uniform Acc5 edges (width 6/5):")
    edges = uniform5_edges()
    for i, (lo, hi) in enumerate(zip(edges, edges[1:]), start=1):
        print(f"  class {i}: [{lo:.6f}, {hi:.6f})")
    print()


def _bin_examples() -> None:
    samples = [-3.0, -2.2, -1.0, -0.4, 0.0, 0.4, 0.5, 1.5, 2.9, 3.0]
    pred7 = split_uniform_7(samples)
    pred5 = split_uniform_5(samples)
    print("Digitize walkthrough:")
    print(f"{'value':>8}  {'Acc7':>4}  {'Acc5':>4}")
    for value, c7, c5 in zip(samples, pred7, pred5):
        print(f"{value:8.2f}  {int(c7):4d}  {int(c5):4d}")
    print()
    # 0.4 and 0.5 sit on opposite sides of the Acc7 boundary at ≈0.429.
    if pred7[5] == pred7[6]:
        raise SystemExit("expected 0.4 and 0.5 to land in different Acc7 bins")
    print("Note: 0.40 and 0.50 are both near-neutral but fall in different Acc7 bins.")
    print()


def _metric_examples() -> None:
    truth = [-2.0, -0.2, 0.0, 0.8, 2.4]
    pred = [-1.5, 0.3, 0.1, 0.9, 2.0]
    reg = regression_scores(truth, pred)
    clf = classification_scores(truth, pred)
    f1, acc2 = acc2_f1(truth, pred)

    print("Hand-checked five-point example")
    print(f"  truth = {truth}")
    print(f"  pred  = {pred}")
    print(f"  MAE   = {reg['MAE']:.4f}   (mean |err| = (0.5+0.5+0.1+0.1+0.4)/5 = 0.32)")
    print(f"  MSE   = {reg['MSE']:.4f}")
    print(f"  Corr  = {reg['Corr']:.4f}")
    print(f"  Acc7  = {clf['Acc7_uniform']:.4f}")
    print(f"  Acc5  = {clf['Acc5_uniform']:.4f}")
    print(f"  Acc2  = {acc2:.4f}   (gold 0.0 dropped; signs compared with > 0)")
    print(f"  F1    = {f1:.4f}")
    print()

    expected_mae = 0.32
    if abs(reg["MAE"] - expected_mae) > 1e-9:
        raise SystemExit(f"MAE mismatch: {reg['MAE']} != {expected_mae}")

    # Drop the gold-zero item: remaining signs truth -, -, +, + vs pred +, +, +, +
    # Wait: truth [-2, -0.2, 0.8, 2.4] → F F T T
    # pred  [-1.5, 0.3, 0.9, 2.0] → F T T T
    # accuracy = 3/4 = 0.75
    if abs(acc2 - 0.75) > 1e-9:
        raise SystemExit(f"Acc2 mismatch: {acc2} != 0.75")
    print("Checks passed: MAE is 0.32 and Acc2 is 0.75 after dropping the neutral gold.")


def main() -> int:
    _print_edges()
    _bin_examples()
    _metric_examples()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
