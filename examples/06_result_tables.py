#!/usr/bin/env python3
"""Pretty-print the checked-in personal CSVs and rank methods by MAE.

This is the command-line companion to ``docs/experiments-and-results.md``.
It only reads files already in git; it does not train or download data.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "model" / "results"
MOSI = ROOT / "model" / "mosi_test"

TABLES = [
    ("MOSEI BERT fusion", RESULTS / "main_results.csv"),
    ("MOSEI GloVe fusion", RESULTS / "glove_results.csv"),
    ("MOSEI BERT GMTM ablation", RESULTS / "ablation_results.csv"),
    ("MOSEI GloVe GMTM ablation", RESULTS / "ablation_glove_results.csv"),
    ("MOSI-pool BERT transfer", MOSI / "mosi_bert_results.csv"),
    ("MOSI-pool GloVe transfer", MOSI / "mosi_glove_results.csv"),
    ("MOSI-pool BERT GMTM ablation", MOSI / "ablation_mosi_results.csv"),
    ("MOSI-pool GloVe GMTM ablation", MOSI / "ablation_mosi_glove_results.csv"),
]


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(row: dict[str, str], key: str) -> str:
    # Column names differ slightly (ACC7 vs Acc5 vs ACC2).
    aliases = {
        "ACC7": ("ACC7", "Acc7"),
        "Acc5": ("Acc5", "ACC5"),
        "ACC2": ("ACC2", "Acc2"),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in row and row[candidate] != "":
            return f"{float(row[candidate]):.4f}"
    return "  n/a"


def show(title: str, path: Path) -> None:
    rows = load(path)
    ranked = sorted(rows, key=lambda r: float(r["MAE"]))
    print(f"=== {title} ===")
    print(f"    {path.relative_to(ROOT)}")
    print(f"    {'method':<32} {'MAE':>7} {'Acc7':>7} {'Acc2':>7} {'Corr':>7} {'F1':>7}")
    for row in ranked:
        name = row["Fusion Method"]
        print(
            f"    {name:<32} {float(row['MAE']):7.4f} "
            f"{fmt(row, 'ACC7'):>7} {fmt(row, 'ACC2'):>7} "
            f"{float(row['Corr']):>+7.4f} {float(row['F1']):7.4f}"
        )
    best = ranked[0]
    print(f"    lowest MAE: {best['Fusion Method']} ({float(best['MAE']):.4f})")
    print()


def main() -> None:
    missing = [path for _, path in TABLES if not path.is_file()]
    if missing:
        raise SystemExit("missing CSVs:\n" + "\n".join(str(p) for p in missing))
    for title, path in TABLES:
        show(title, path)
    print("MOSI-pool tables use the merged train+valid+test loader; see docs/datasets.md.")


if __name__ == "__main__":
    main()
