#!/usr/bin/env python3
"""Run the six numbered example scripts in order from the repo root."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ROOT / "examples" / "01_synthetic_batch.py",
    ROOT / "examples" / "02_fusion_forward.py",
    ROOT / "examples" / "03_gated_transformer_toy.py",
    ROOT / "examples" / "04_metrics_walkthrough.py",
    ROOT / "examples" / "05_packed_vs_padded.py",
    ROOT / "examples" / "06_ablation_zero_modalities.py",
]


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    print(f"repo root: {ROOT}")
    for path in EXAMPLES:
        print("\n" + "=" * 72)
        print(f"running {path.relative_to(ROOT)}")
        print("=" * 72)
        runpy.run_path(str(path), run_name="__main__")
    print("\nall examples finished")


if __name__ == "__main__":
    main()
