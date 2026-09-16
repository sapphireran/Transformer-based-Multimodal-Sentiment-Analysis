#!/usr/bin/env python3
"""Run the personal example suite and summarize pass/fail."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES = [
    ("summarize_results.py", False),
    ("metric_walkthrough.py", False),
    ("tiny_affect_dataset.py", False),
    ("forward_pass_demo.py", True),
    ("fusion_shape_walkthrough.py", True),
    ("toy_train_gmtm.py", True),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Only run examples that do not import PyTorch",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    results = []
    for name, needs_torch in EXAMPLES:
        if needs_torch and args.skip_torch:
            results.append((name, "skip", 0.0, ""))
            continue
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(here / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.time() - started
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        snippet = tail[-1] if tail else ""
        results.append((name, status, elapsed, snippet))
        print(f"[{status:8s} {elapsed:6.2f}s] {name}")
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)

    print()
    failed = [name for name, status, _, _ in results if status.startswith("fail")]
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    print("all requested examples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
