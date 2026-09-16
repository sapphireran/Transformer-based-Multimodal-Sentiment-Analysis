#!/usr/bin/env python3
"""Run every example script in a fixed order and fail on the first error."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from paths import EXAMPLES_DIR

SCRIPTS = [
    "run_metrics.py",
    "run_results_table.py",
    "plot_published_results.py",
    "run_fusion_forward.py",
    "run_gmtm_toy.py",
]


def main() -> int:
    python = sys.executable
    for name in SCRIPTS:
        path = EXAMPLES_DIR / name
        print(f"\n=== {name} ===")
        proc = subprocess.run([python, str(path)], cwd=str(EXAMPLES_DIR))
        if proc.returncode != 0:
            print(f"{name} failed with {proc.returncode}")
            return proc.returncode
    print("\nall example scripts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
