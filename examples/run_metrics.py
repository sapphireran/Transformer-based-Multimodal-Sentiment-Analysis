#!/usr/bin/env python3
"""Show the uniform-bin and binary metrics on a controlled synthetic set.

Prints the Acc-7 / Acc-5 edges from docs/evaluation.md and scores:
  * a perfect copy of y
  * y plus noise
  * a sign-flipped copy (should crush Acc-2 / F1 / Corr)
"""

from __future__ import annotations

import json

import numpy as np

from metrics_lib import (
    evaluate_affect_batch,
    format_metrics,
    split_uniform_5,
    split_uniform_7,
    uniform_edges,
)
from paths import ensure_output_dir


def _demo_pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    y = np.array([-2.8, -1.7, -0.6, 0.0, 0.2, 1.1, 2.4, 2.9], dtype=np.float64)
    perfect = y.copy()
    noisy = y + rng.normal(0.0, 0.35, size=y.shape)
    flipped = -y
    constant = np.zeros_like(y)
    return {
        "perfect": (y, perfect),
        "noisy_sigma_0.35": (y, noisy),
        "sign_flipped": (y, flipped),
        "always_zero": (y, constant),
    }


def main() -> int:
    print("Acc-7 edges:", np.round(uniform_edges(7), 3).tolist())
    print("Acc-5 edges:", np.round(uniform_edges(5), 3).tolist())
    y_grid = np.array([-3.0, -2.2, -0.4, 0.0, 0.4, 1.3, 3.0])
    print("sample → bin7:", list(zip(y_grid.tolist(), split_uniform_7(y_grid).tolist())))
    print("sample → bin5:", list(zip(y_grid.tolist(), split_uniform_5(y_grid).tolist())))
    print()

    report = {"edges7": uniform_edges(7).tolist(), "edges5": uniform_edges(5).tolist(), "cases": {}}
    for name, (y, yhat) in _demo_pairs().items():
        row = evaluate_affect_batch(y, yhat)
        report["cases"][name] = row
        print(f"{name:18s}  {format_metrics(row)}")

    # sanity: perfect copy is exact on every discrete metric
    perfect = report["cases"]["perfect"]
    assert perfect["MAE"] == 0.0
    assert perfect["Acc7_uniform"] == 1.0
    assert perfect["Acc5_uniform"] == 1.0
    assert perfect["Acc2"] == 1.0
    assert perfect["F1"] == 1.0
    assert perfect["Corr"] > 0.999

    flipped = report["cases"]["sign_flipped"]
    assert flipped["Corr"] < 0.0
    assert flipped["Acc2"] < 0.5

    path = ensure_output_dir() / "metrics_demo.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
