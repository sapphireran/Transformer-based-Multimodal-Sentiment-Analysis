"""Pretty-print the CSV tables already checked into ``model/results``.

No torch, no pickles. Useful when you want the personal experiment log
on a machine that cannot train.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

from common import ROOT

RESULTS = ROOT / "model" / "results"
MOSI_TEST = ROOT / "model" / "mosi_test"

TABLES = [
    ("MOSEI BERT fusion", RESULTS / "main_results.csv"),
    ("MOSEI GloVe fusion", RESULTS / "glove_results.csv"),
    ("MOSEI BERT GMTM ablation", RESULTS / "ablation_results.csv"),
    ("MOSEI GloVe GMTM ablation", RESULTS / "ablation_glove_results.csv"),
    ("MOSI transfer BERT", MOSI_TEST / "mosi_bert_results.csv"),
    ("MOSI transfer GloVe", MOSI_TEST / "mosi_glove_results.csv"),
    ("MOSI GMTM ablation BERT", MOSI_TEST / "ablation_mosi_results.csv"),
    ("MOSI GMTM ablation GloVe", MOSI_TEST / "ablation_mosi_glove_results.csv"),
]


def load_table(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _widths(rows: List[List[str]]) -> List[int]:
    cols = max(len(r) for r in rows)
    widths = [0] * cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def format_table(rows: List[List[str]]) -> str:
    if not rows:
        return "(empty)"
    widths = _widths(rows)
    lines = []
    for idx, row in enumerate(rows):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        # Right-align numeric columns after the name column.
        for i, cell in enumerate(row):
            if i == 0:
                continue
            padded[i] = cell.rjust(widths[i])
        lines.append("  ".join(padded))
        if idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def best_mae_row(rows: List[List[str]]) -> List[str]:
    header, *body = rows
    try:
        mae_idx = header.index("MAE")
    except ValueError:
        return body[0]
    return min(body, key=lambda r: float(r[mae_idx]))


def run(verbose: bool = True) -> List[dict]:
    summaries = []
    for title, path in TABLES:
        if not path.exists():
            summaries.append({"title": title, "path": str(path), "ok": False})
            if verbose:
                print(f"\n## {title}\nmissing: {path}")
            continue
        rows = load_table(path)
        winner = best_mae_row(rows)
        summaries.append(
            {
                "title": title,
                "path": str(path),
                "ok": True,
                "n": len(rows) - 1,
                "best": winner[0],
                "best_mae": winner[1],
            }
        )
        if verbose:
            print()
            print(f"## {title}")
            print(f"file: {path.relative_to(ROOT)}")
            print(format_table(rows))
            print(f"lowest MAE: {winner[0]}  ({winner[1]})")
    return summaries


def main() -> int:
    rows = run(verbose=True)
    missing = [r["title"] for r in rows if not r["ok"]]
    if missing:
        print("missing tables:", ", ".join(missing), file=sys.stderr)
        return 1
    print()
    print(f"logged-results printer passed ({len(rows)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
