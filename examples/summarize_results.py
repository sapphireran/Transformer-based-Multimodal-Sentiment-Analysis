#!/usr/bin/env python3
"""Reprint every checked-in experiment CSV with rank columns.

No PyTorch required. This is the fastest way to read the personal logs
without opening a notebook.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Allow `python examples/summarize_results.py` without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.repo import RESULT_CSV_PATHS

LOWER_IS_BETTER = {"MAE", "MSE"}
HIGHER_IS_BETTER = {"ACC7", "ACC5", "ACC2", "CORR", "F1", "ACC7_UNIFORM", "ACC5_UNIFORM"}


def _norm(name: str) -> str:
    return name.strip().upper().replace(" ", "")


def read_table(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value.strip().replace("−", "-"))


def rank_rows(rows: Sequence[Dict[str, str]], metric: str) -> List[int]:
    key = None
    for candidate in rows[0].keys():
        if _norm(candidate) == _norm(metric):
            key = candidate
            break
    if key is None:
        return [0] * len(rows)

    scores = [_float(row[key]) for row in rows]
    reverse = _norm(metric) in HIGHER_IS_BETTER
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=reverse)
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def format_table(title: str, path: Path) -> str:
    rows = read_table(path)
    if not rows:
        return f"## {title}\n\n(empty) `{path}`\n"

    headers = list(rows[0].keys())
    metric_headers = [name for name in headers[1:]]
    ranks_by_metric = {name: rank_rows(rows, name) for name in metric_headers}

    col_w = [max(len(h), 18) for h in headers]
    for row in rows:
        for i, name in enumerate(headers):
            col_w[i] = max(col_w[i], len(str(row[name])))

    lines = [
        f"## {title}",
        f"",
        f"Source: `{path.as_posix()}`",
        f"",
    ]
    header_line = " | ".join(name.ljust(col_w[i]) for i, name in enumerate(headers))
    rule = "-|-".join("-" * col_w[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(rule)
    for row in rows:
        lines.append(" | ".join(str(row[name]).ljust(col_w[i]) for i, name in enumerate(headers)))

    lines.append("")
    lines.append("Ranks (1 = best for that column):")
    rank_header = " | ".join(["method".ljust(col_w[0])] + [h.ljust(8) for h in metric_headers])
    lines.append(rank_header)
    for i, row in enumerate(rows):
        cells = [str(row[headers[0]]).ljust(col_w[0])]
        for name in metric_headers:
            cells.append(str(ranks_by_metric[name][i]).ljust(8))
        lines.append(" | ".join(cells))

    best = []
    for name in metric_headers:
        winner = min(range(len(rows)), key=lambda i: ranks_by_metric[name][i])
        best.append(f"{name}={rows[winner][headers[0]]}")
    lines.append("")
    lines.append("Best per column: " + "; ".join(best))
    lines.append("")
    return "\n".join(lines)


TITLES = {
    "mosei_bert_fusion": "MOSEI BERT — classical fusion",
    "mosei_glove_fusion": "MOSEI GloVe — classical fusion",
    "mosei_bert_gmtm": "MOSEI BERT — GMTM modality ablation",
    "mosei_glove_gmtm": "MOSEI GloVe — GMTM modality ablation",
    "mosi_bert_fusion": "MOSI transfer BERT — classical fusion (merged splits)",
    "mosi_glove_fusion": "MOSI transfer GloVe — classical fusion (merged splits)",
    "mosi_bert_gmtm": "MOSI transfer BERT — GMTM ablation (merged splits)",
    "mosi_glove_gmtm": "MOSI transfer GloVe — GMTM ablation (merged splits)",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(RESULT_CSV_PATHS),
        help="Subset of table keys to print",
    )
    args = parser.parse_args(argv)

    keys = args.only or list(TITLES)
    chunks = ["# Personal experiment CSVs", ""]
    for key in keys:
        path = RESULT_CSV_PATHS[key]
        if not path.exists():
            chunks.append(f"## {TITLES[key]}\n\nMissing: `{path}`\n")
            continue
        chunks.append(format_table(TITLES[key], path))
    text = "\n".join(chunks)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
