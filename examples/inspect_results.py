#!/usr/bin/env python3
"""Pretty-print the committed MOSI / MOSEI CSV tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_DIR, REPO_ROOT  # noqa: E402

TABLES = (
    ("MOSEI BERT fusion zoo", MODEL_DIR / "main_results.csv"),
    ("MOSEI BERT GMTM ablation", MODEL_DIR / "ablation_results.csv"),
    ("MOSEI GloVe fusion zoo", MODEL_DIR / "glove_results.csv"),
    ("MOSEI GloVe GMTM ablation", MODEL_DIR / "ablation_glove_results.csv"),
    ("MOSI BERT transfer", MODEL_DIR / "mosi_test" / "mosi_bert_results.csv"),
    ("MOSI GloVe transfer", MODEL_DIR / "mosi_test" / "mosi_glove_results.csv"),
    ("MOSI BERT GMTM ablation", MODEL_DIR / "mosi_test" / "ablation_mosi_results.csv"),
    ("MOSI GloVe GMTM ablation", MODEL_DIR / "mosi_test" / "ablation_mosi_glove_results.csv"),
)

NUMERIC_HIGHER_BETTER = {"ACC7", "ACC5", "ACC2", "CORR", "F1"}
NUMERIC_LOWER_BETTER = {"MAE"}


def _normalize_header(name: str) -> str:
    return name.strip()


def _clean_method(value: str) -> str:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text.strip("[]")
        parts = [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
        return "+".join(parts)
    return text


def _to_float(value: str) -> float:
    return float(value)


def load_table(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {_normalize_header(k): v for k, v in raw.items() if k}
            if "Fusion Method" in row:
                row["Fusion Method"] = _clean_method(row["Fusion Method"])
            rows.append(row)
        return rows


def _metric_keys(rows: Sequence[Dict[str, str]]) -> List[str]:
    keys = []
    for key in rows[0].keys():
        if key == "Fusion Method":
            continue
        keys.append(key)
    return keys


def _best_flags(rows: Sequence[Dict[str, str]], keys: Sequence[str]) -> Dict[str, str]:
    winners = {}
    for key in keys:
        numeric = []
        for row in rows:
            try:
                numeric.append((row["Fusion Method"], _to_float(row[key])))
            except (TypeError, ValueError):
                continue
        if not numeric:
            continue
        upper = key.upper()
        if any(token in upper for token in NUMERIC_LOWER_BETTER):
            name, _ = min(numeric, key=lambda item: item[1])
        else:
            name, _ = max(numeric, key=lambda item: item[1])
        winners[key] = name
    return winners


def render(title: str, path: Path) -> str:
    if not path.exists():
        return f"## {title}\n\nmissing: {path.relative_to(REPO_ROOT)}\n"
    rows = load_table(path)
    if not rows:
        return f"## {title}\n\nempty: {path.relative_to(REPO_ROOT)}\n"
    keys = _metric_keys(rows)
    winners = _best_flags(rows, keys)
    header = ["Method", *keys]
    body = []
    for row in rows:
        cells = [row["Fusion Method"]]
        for key in keys:
            raw = row[key]
            try:
                number = f"{_to_float(raw):.4f}"
            except ValueError:
                number = raw
            if winners.get(key) == row["Fusion Method"]:
                number = f"*{number}*"
            cells.append(number)
        body.append(cells)
    widths = [len(col) for col in header]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [
        f"## {title}",
        f"source: {path.relative_to(REPO_ROOT)}",
        "",
        fmt(header),
        fmt(["-" * w for w in widths]),
    ]
    lines.extend(fmt(cells) for cells in body)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="substring filter on table title")
    args = parser.parse_args(argv)

    printed = 0
    for title, path in TABLES:
        if args.only and args.only.lower() not in title.lower():
            continue
        print(render(title, path))
        printed += 1
    if printed == 0:
        print("no tables matched")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
