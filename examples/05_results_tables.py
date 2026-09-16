#!/usr/bin/env python3
"""Reprint the checked-in MOSI / MOSEI CSVs as Markdown tables.

    python examples/05_results_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from common import MODEL_DIR, REPO_ROOT

TABLES = [
    (
        "MOSEI + BERT — fusion baselines",
        MODEL_DIR / "results" / "main_results.csv",
    ),
    (
        "MOSEI + BERT — GMTM ablation",
        MODEL_DIR / "results" / "ablation_results.csv",
    ),
    (
        "MOSEI + GloVe — fusion baselines",
        MODEL_DIR / "results" / "glove_results.csv",
    ),
    (
        "MOSEI + GloVe — GMTM ablation",
        MODEL_DIR / "results" / "ablation_glove_results.csv",
    ),
    (
        "MOSI transfer + BERT — fusion + GMTM",
        MODEL_DIR / "mosi_test" / "mosi_bert_results.csv",
    ),
    (
        "MOSI transfer + BERT — GMTM ablation",
        MODEL_DIR / "mosi_test" / "ablation_mosi_results.csv",
    ),
    (
        "MOSI transfer + GloVe — fusion + GMTM",
        MODEL_DIR / "mosi_test" / "mosi_glove_results.csv",
    ),
    (
        "MOSI transfer + GloVe — GMTM ablation",
        MODEL_DIR / "mosi_test" / "ablation_mosi_glove_results.csv",
    ),
]


def read_csv(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0], rows[1:]


def as_markdown(header, rows) -> str:
    # Normalise the slightly different Acc5 / ACC5 spellings in the files.
    pretty = []
    for cell in header:
        pretty.append(cell.replace("ACC7", "Acc7").replace("ACC5", "Acc5").replace("ACC2", "Acc2"))
    lines = [
        "| " + " | ".join(pretty) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(pretty))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    print(f"Reading CSVs under {MODEL_DIR.relative_to(REPO_ROOT)}")
    print()
    missing = []
    for title, path in TABLES:
        if not path.exists():
            missing.append(path)
            print(f"## {title}")
            print(f"missing: {path}")
            print()
            continue
        header, rows = read_csv(path)
        print(f"## {title}")
        print()
        print(f"_{path.relative_to(REPO_ROOT)}, {len(rows)} rows_")
        print()
        print(as_markdown(header, rows))
        print()
    if missing:
        return 1
    print("All eight result CSVs found. Narrative: docs/experiments.md and docs/mosi.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
