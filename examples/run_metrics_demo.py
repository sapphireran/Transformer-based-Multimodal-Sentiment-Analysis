#!/usr/bin/env python3
"""Score a synthetic (gold, prediction) pair with the repo protocol.

Three slices are printed:

1. A near-perfect predictor (prediction = gold + tiny noise) so you
   can see every metric approach its ceiling.
2. A shuffled predictor, which should collapse Acc-2 toward chance
   and Pearson toward 0.
3. A constant predictor, which is the one case Pearson is undefined.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

from examples.metrics import evaluate_regression, split_uniform_5, split_uniform_7


def _bounded_scores(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 3.0 * np.tanh(rng.normal(size=n))


def _print_report(title: str, report) -> None:
    print(f"\n== {title} ==")
    print(report.pretty())
    print("as_dict:", {k: round(v, 4) if v == v else v for k, v in report.as_dict().items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = _bounded_scores(args.n, args.seed)
    print(f"n={args.n}  gold range [{gold.min():+.3f}, {gold.max():+.3f}]")
    print(
        f"Acc-7 bin occupancy (gold): {dict(zip(*np.unique(split_uniform_7(gold), return_counts=True)))}"
    )
    print(
        f"Acc-5 bin occupancy (gold): {dict(zip(*np.unique(split_uniform_5(gold), return_counts=True)))}"
    )

    rng = np.random.default_rng(args.seed + 1)
    near = np.clip(gold + rng.normal(scale=0.15, size=args.n), -3.0, 3.0)
    _print_report("near-perfect predictor (noise 0.15)", evaluate_regression(gold, near))

    shuffled = gold.copy()
    rng.shuffle(shuffled)
    _print_report("shuffled predictor", evaluate_regression(gold, shuffled))

    constant = np.full(args.n, gold.mean())
    const_report = evaluate_regression(gold, constant)
    _print_report("constant predictor (Corr should be nan)", const_report)
    if not np.isnan(const_report.corr):
        raise SystemExit("expected Pearson r to be nan for a constant predictor")

    # Guard the protocol itself: equal-width Acc-7 must have 7 bins, and
    # a perfect prediction must get Acc-7 = Acc-5 = 1.
    perfect = evaluate_regression(gold, gold)
    if perfect.acc7 != 1.0 or perfect.acc5 != 1.0 or perfect.mae != 0.0:
        raise SystemExit(f"perfect predictor failed the protocol: {perfect.as_dict()}")
    print("\nProtocol checks passed (perfect Acc-7/Acc-5 = 1, constant Corr is nan).")


if __name__ == "__main__":
    main()
