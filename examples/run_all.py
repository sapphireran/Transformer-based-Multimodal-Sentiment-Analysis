#!/usr/bin/env python3
"""Run the CPU example suite in a fixed order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent

STEPS = (
    ("inspect committed CSVs", [sys.executable, str(EXAMPLES / "inspect_results.py"), "--only", "MOSEI BERT"]),
    ("score toy labels", [sys.executable, str(EXAMPLES / "evaluate_toy.py"), "--write-json"]),
    ("fusion zoo forward", [sys.executable, str(EXAMPLES / "fusion_zoo.py"), "--write-json"]),
    ("GMTM forward", [sys.executable, str(EXAMPLES / "gmtm_forward.py"), "--write-json"]),
    ("GMTM toy train", [sys.executable, str(EXAMPLES / "train_toy_gmtm.py"), "--epochs", "4", "--write-json"]),
    ("toy modality ablation", [sys.executable, str(EXAMPLES / "ablate_toy_modalities.py"), "--write-json"]),
)


def main() -> int:
    failures = 0
    for title, command in STEPS:
        print("=" * 72)
        print(title)
        print(" ".join(command))
        print("=" * 72)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"FAILED ({result.returncode}): {title}")
            failures += 1
        print()
    if failures:
        print(f"{failures} example step(s) failed")
        return 1
    print("all example steps passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
