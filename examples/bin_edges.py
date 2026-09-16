#!/usr/bin/env python3
"""Print the uniform Acc-7 / Acc-5 bin edges and classify boundary scores.

This is the definition behind every ``Acc7_uniform`` / ``Acc5_uniform`` cell
in the CSVs. ``np.digitize(..., right=False)`` uses half-open intervals
``[left, right)``, then clips so ``±3`` land in the first / last bin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402

add_model_to_path()

from metrics import bin_intervals, split_uniform, uniform_bin_edges  # noqa: E402

PROBES = (-3.0, -2.999, -2.0, -1.0, 0.0, 1.0, 2.0, 2.999, 3.0, 3.1, -3.1)


def _table(n_bins: int) -> str:
    lines = [f"Acc-{n_bins}  width={(6.0 / n_bins):.6f}", "idx  left      right     (interval)"]
    for idx, left, right in bin_intervals(n_bins):
        closed = "]" if idx == n_bins else ")"
        lines.append(f"{idx:>3}  {left:+.6f}  {right:+.6f}  [{left:+.3f}, {right:+.3f}{closed}")
    return "\n".join(lines)


def _probe_block(n_bins: int) -> str:
    values = list(PROBES)
    bins = split_uniform(values, n_bins=n_bins)
    lines = [f"score → Acc-{n_bins} bin"]
    for score, bucket in zip(values, bins):
        lines.append(f"  {score:+6.3f}  →  {int(bucket)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    print(_table(7))
    print()
    print(_probe_block(7))
    print()
    print(_table(5))
    print()
    print(_probe_block(5))

    if args.write_json:
        payload = {
            "acc7_edges": uniform_bin_edges(7).tolist(),
            "acc5_edges": uniform_bin_edges(5).tolist(),
            "probes": {
                str(score): {"acc7": int(split_uniform([score], 7)[0]), "acc5": int(split_uniform([score], 5)[0])}
                for score in PROBES
            },
        }
        out = ensure_output_dir() / "bin_edges.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
