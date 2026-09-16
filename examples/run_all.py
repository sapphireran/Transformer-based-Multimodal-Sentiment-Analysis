#!/usr/bin/env python3
"""Run the four CPU demos in order. Exits non-zero on the first failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from examples.repo import ROOT

DEMOS = (
    "examples/run_fusion_demo.py",
    "examples/run_gmtm_demo.py",
    "examples/run_metrics_demo.py",
    "examples/run_tiny_train.py",
)


def main() -> int:
    for rel in DEMOS:
        path = ROOT / rel
        print(f"\n######## {rel} ########\n")
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n{rel} failed with exit {result.returncode}")
            return result.returncode
    print("\nAll demos exited 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
