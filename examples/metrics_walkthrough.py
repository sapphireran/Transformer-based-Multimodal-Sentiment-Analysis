#!/usr/bin/env python3
"""Walk through the repo's MOSI/MOSEI metric definitions on a tiny vector.

Shows the uniform 5/7-bin edges, which neutrals Acc-2 drops, and the full
`summarize_predictions` dict. Numbers are deterministic (hard-coded y / yhat).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.metrics import (
    binary_acc_f1,
    split_uniform_5,
    split_uniform_7,
    summarize_predictions,
    uniform_edges,
)


# Hand-picked so several bins, one exact zero, and one sign error are visible.
Y = [-3.0, -2.1, -1.0, -0.2, 0.0, 0.0, 0.4, 1.2, 1.8, 2.4, 2.9, 3.0]
YHAT = [-2.6, -1.4, -1.3, 0.3, 0.1, -0.4, 0.5, 0.9, 2.2, 1.7, 2.5, 2.8]


def _fmt_edges(n: int) -> str:
    edges = uniform_edges(n)
    parts = [f"[{edges[i]:.4f}, {edges[i+1]:.4f})" for i in range(n)]
    parts[-1] = parts[-1][:-1] + "]"
    return f"Acc-{n} edges: " + ", ".join(parts)


def main() -> int:
    print("gold y :", " ".join(f"{v:5.1f}" for v in Y))
    print("pred ŷ :", " ".join(f"{v:5.1f}" for v in YHAT))
    print()
    print(_fmt_edges(7))
    print("y bins7:", [int(v) for v in split_uniform_7(Y)])
    print("ŷ bins7:", [int(v) for v in split_uniform_7(YHAT)])
    print()
    print(_fmt_edges(5))
    print("y bins5:", [int(v) for v in split_uniform_5(Y)])
    print("ŷ bins5:", [int(v) for v in split_uniform_5(YHAT)])
    print()

    f1_all, acc_all = binary_acc_f1(Y, YHAT, exclude_zero=False)
    f1, acc = binary_acc_f1(Y, YHAT, exclude_zero=True)
    print(f"Acc-2 including neutrals (y==0 kept): acc={acc_all:.4f} f1={f1_all:.4f}")
    print(f"Acc-2 exclude_zero (matches eval_affect): acc={acc:.4f} f1={f1:.4f}")
    print("  dropped gold zeros at indices", [i for i, v in enumerate(Y) if v == 0.0])
    print("  sign error at y=-0.2 vs ŷ=+0.3 (counts against Acc-2)")
    print()

    summary = summarize_predictions(Y, YHAT)
    print("summarize_predictions:")
    for key in ("MAE", "MSE", "Corr", "Acc7_uniform", "Acc5_uniform", "Acc2", "F1"):
        print(f"  {key:14s} {summary[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
