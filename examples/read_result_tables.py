#!/usr/bin/env python3
"""Pretty-print the recorded experiment CSVs without importing torch models."""

from __future__ import annotations

import csv
from pathlib import Path

from common import MOSI_RESULTS_DIR, REPO_ROOT, RESULTS_DIR

TABLES = [
    (RESULTS_DIR / "main_results.csv", "MOSEI · BERT baselines"),
    (RESULTS_DIR / "glove_results.csv", "MOSEI · GloVe baselines"),
    (RESULTS_DIR / "ablation_results.csv", "MOSEI · BERT GMTM ablation"),
    (RESULTS_DIR / "ablation_glove_results.csv", "MOSEI · GloVe GMTM ablation"),
    (MOSI_RESULTS_DIR / "mosi_bert_results.csv", "MOSI transfer · BERT (merged splits)"),
    (MOSI_RESULTS_DIR / "mosi_glove_results.csv", "MOSI transfer · GloVe (merged splits)"),
    (MOSI_RESULTS_DIR / "ablation_mosi_results.csv", "MOSI transfer · BERT GMTM ablation"),
    (MOSI_RESULTS_DIR / "ablation_mosi_glove_results.csv", "MOSI transfer · GloVe GMTM ablation"),
]


def _print_table(path: Path, title: str) -> None:
    if not path.is_file():
        print(f"\n## {title}\n  missing: {path}")
        return
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        print(f"\n## {title}\n  empty: {path}")
        return
    widths = [max(len(row[c]) if c < len(row) else 0 for row in rows) for c in range(len(rows[0]))]
    print(f"\n## {title}")
    print(f"   {path.relative_to(REPO_ROOT)}")
    for i, row in enumerate(rows):
        cells = [row[c].ljust(widths[c]) if c < len(row) else "".ljust(widths[c]) for c in range(len(widths))]
        line = "  ".join(cells)
        print(line)
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    print("Recorded tables (copied, not re-fit). MOSI rows use merged train+valid+test.")
    for path, title in TABLES:
        _print_table(path, title)
    print(
        "\nBest in-domain MOSEI BERT row in this log: GMTM text+audio+visual"
        " (MAE 0.5640, Acc-2 0.8429, F1 0.8777)."
    )


if __name__ == "__main__":
    main()
