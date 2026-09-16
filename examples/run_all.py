"""Run the data-free personal examples and fail if any demo breaks."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python examples/run_all.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ablation_deltas
import fusion_shapes
import gmtm_forward
import inspect_pickle
import metrics_walkthrough
import summarize_logged_results
import synthetic_batch
import tiny_overfit


def main() -> None:
    steps = [
        ("synthetic batch", synthetic_batch._demo),
        ("metrics walkthrough", metrics_walkthrough._demo),
        ("GMTM forward", gmtm_forward._demo),
        ("fusion shapes", fusion_shapes._demo),
        ("logged CSVs", summarize_logged_results._demo),
        ("ablation deltas", ablation_deltas._demo),
        ("tiny overfit", tiny_overfit._demo),
        ("pickle inspect (optional)", inspect_pickle._demo),
    ]
    print("=" * 60)
    print("personal examples — CPU, no MOSI/MOSEI download required")
    print("=" * 60)
    for name, fn in steps:
        print()
        print(f"--- {name} ---")
        fn()
        print(f"ok  {name}")
    print()
    print("all examples finished")


if __name__ == "__main__":
    main()
