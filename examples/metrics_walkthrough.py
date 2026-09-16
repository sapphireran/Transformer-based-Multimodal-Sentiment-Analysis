"""Walk through MAE / Acc-2 / Acc-5 / Acc-7 / F1 on a tiny labeled set."""

from __future__ import annotations

import numpy as np

from examples.eval_metrics import (
    bin_edges_5,
    bin_edges_7,
    evaluate,
    round_row,
    split_uniform_5,
    split_uniform_7,
)


# Chosen so one gold clip is exactly 0 (dropped from Acc-2) and one pair
# disagrees in sign (the -0.2 vs 0.4 pair).
GOLD = np.array([-2.4, -0.2, 0.0, 1.1, 2.8])
PRED = np.array([-2.1, 0.4, 0.1, 0.8, 2.2])


def _bin_table(values: np.ndarray, kind: str) -> list[tuple[str, int]]:
    if kind == "7":
        bins = split_uniform_7(values)
        edges = bin_edges_7()
    else:
        bins = split_uniform_5(values)
        edges = bin_edges_5()
    rows = []
    for v, b in zip(values, bins):
        lo, hi = edges[b - 1], edges[b]
        rows.append((f"{v:+.2f} in [{lo:.3f}, {hi:.3f})", int(b)))
    return rows


def main() -> int:
    print("Gold :", GOLD.tolist())
    print("Pred :", PRED.tolist())
    print()
    print("Uniform 7-bin assignment (gold)")
    for label, bucket in _bin_table(GOLD, "7"):
        print(f"  {label:<28} -> class {bucket}")
    print("Uniform 5-bin assignment (gold)")
    for label, bucket in _bin_table(GOLD, "5"):
        print(f"  {label:<28} -> class {bucket}")

    metrics = evaluate(GOLD, PRED)
    print()
    print("Metric dict (same keys as single_test, minus TestLoss)")
    for key, value in round_row(metrics).items():
        print(f"  {key:<14} {value}")

    # Hand checks that tests also assert.
    mae = float(np.mean(np.abs(GOLD - PRED)))
    if abs(metrics["MAE"] - mae) > 1e-9:
        raise RuntimeError("MAE mismatch")
    # Acc-2 drops the 0.0 gold clip; remaining signs: - - + + vs - + + +
    if abs(metrics["Acc2"] - 0.75) > 1e-9:
        raise RuntimeError(f"expected Acc-2 0.75, got {metrics['Acc2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
