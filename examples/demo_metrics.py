"""Walk through the evaluation protocol on hand-picked scores.

Prints the equal-width Acc7 / Acc5 edges and shows how a few (y, ŷ)
pairs land. Also compares a perfect predictor, a sign-only predictor,
and a constant-zero predictor — the last one is the usual reminder that
Acc2 *excludes* gold zeros and then tests ``ŷ > 0``.
"""

from __future__ import annotations

import sys

import numpy as np

from eval_protocol import (
    acc5_edges,
    acc7_edges,
    eval_affect,
    format_metrics,
    regression_and_classification,
    split_uniform_5,
    split_uniform_7,
)


def _banner(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def show_edges() -> None:
    _banner("Equal-width bin edges on [-3, 3]")
    e7 = acc7_edges()
    e5 = acc5_edges()
    print("Acc7 edges:", "  ".join(f"{x:+.3f}" for x in e7))
    print("Acc5 edges:", "  ".join(f"{x:+.3f}" for x in e5))
    probes = np.array([-3.0, -2.0, -0.5, 0.0, 0.5, 2.0, 3.0])
    print()
    print("value   Acc7   Acc5")
    for value, b7, b5 in zip(probes, split_uniform_7(probes), split_uniform_5(probes)):
        print(f"{value:+5.1f}    {b7:2d}     {b5:2d}")


def show_toy_pairs() -> dict:
    _banner("Hand-picked (gold, pred) pairs")
    y = np.array([-2.4, -1.0, -0.2, 0.0, 0.0, 0.8, 1.7, 2.6])
    yhat = np.array([-2.1, -0.4, 0.3, 0.1, -0.1, 0.9, 1.1, 2.4])
    print(" y   :", " ".join(f"{v:+5.1f}" for v in y))
    print(" ŷ   :", " ".join(f"{v:+5.1f}" for v in yhat))
    print("Acc7 y:", split_uniform_7(y))
    print("Acc7 ŷ:", split_uniform_7(yhat))
    metrics = regression_and_classification(y, yhat)
    print(format_metrics(metrics))
    return metrics


def show_reference_predictors() -> None:
    _banner("Reference predictors on a 12-clip slice")
    rng = np.random.default_rng(0)
    y = np.array([-2.5, -1.2, -0.4, 0.0, 0.0, 0.3, 0.9, 1.4, 2.2, 2.8, -0.8, 1.1])
    perfect = regression_and_classification(y, y)
    sign_only = regression_and_classification(y, np.sign(y) * 1.5)
    zeros = regression_and_classification(y, np.zeros_like(y))
    shuffled = regression_and_classification(y, rng.permutation(y))
    print("perfect   ", format_metrics(perfect))
    print("sign*1.5  ", format_metrics(sign_only))
    print("all-zero  ", format_metrics(zeros))
    print("shuffled  ", format_metrics(shuffled))

    f1_keep, acc2_keep = eval_affect(y, np.zeros_like(y), exclude_zero=True)
    f1_all, acc2_all = eval_affect(y, np.zeros_like(y), exclude_zero=False)
    print()
    print(f"all-zero Acc2 exclude_zero=True  → {acc2_keep:.3f}  F1={f1_keep:.3f}")
    print(f"all-zero Acc2 exclude_zero=False → {acc2_all:.3f}  F1={f1_all:.3f}")
    print("exclude_zero=True is what the CSVs use.")


def main() -> int:
    show_edges()
    metrics = show_toy_pairs()
    show_reference_predictors()
    # Sanity: the hand-picked pair should beat shuffled-level MAE.
    if metrics["MAE"] > 1.5:
        print("unexpectedly large MAE on the hand-picked example", file=sys.stderr)
        return 1
    print()
    print("metrics demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
