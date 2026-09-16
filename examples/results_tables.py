"""Parse the committed experiment CSVs and print the winners."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from examples.common import MOSI_RESULTS_DIR, RESULTS_DIR, print_kv

# (label, path relative to model/, mae-better-is-lower column name)
TABLES = [
    ("MOSEI BERT baselines", RESULTS_DIR / "main_results.csv"),
    ("MOSEI BERT GMTM ablation", RESULTS_DIR / "ablation_results.csv"),
    ("MOSEI GloVe baselines", RESULTS_DIR / "glove_results.csv"),
    ("MOSEI GloVe GMTM ablation", RESULTS_DIR / "ablation_glove_results.csv"),
    ("MOSI BERT transfer", MOSI_RESULTS_DIR / "mosi_bert_results.csv"),
    ("MOSI BERT GMTM ablation", MOSI_RESULTS_DIR / "ablation_mosi_results.csv"),
    ("MOSI GloVe transfer", MOSI_RESULTS_DIR / "mosi_glove_results.csv"),
    ("MOSI GloVe GMTM ablation", MOSI_RESULTS_DIR / "ablation_mosi_glove_results.csv"),
]


def _as_float(cell: str) -> float:
    return float(cell.strip())


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def best_mae_row(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: _as_float(r["MAE"]))


def format_row(row: dict[str, str]) -> str:
    method = row.get("Fusion Method", "?")
    return (
        f"{method:<32} MAE={row['MAE']}  Acc7={row.get('ACC7', row.get('Acc7', ''))}  "
        f"Acc2={row.get('ACC2', '')}  Corr={row.get('Corr', '')}  F1={row.get('F1', '')}"
    )


def summarize() -> list[tuple[str, dict[str, str]]]:
    winners = []
    for title, path in TABLES:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_table(path)
        if not rows:
            raise RuntimeError(f"empty CSV: {path}")
        winners.append((title, best_mae_row(rows)))
    return winners


def main() -> int:
    print("Lowest-MAE row in each committed CSV")
    winners = summarize()
    for title, row in winners:
        print(f"\n{title}")
        print(f"  {format_row(row)}")

    by_title = {title: row for title, row in winners}
    mosei_gmtm = _as_float(by_title["MOSEI BERT GMTM ablation"]["MAE"])
    mosei_base = _as_float(by_title["MOSEI BERT baselines"]["MAE"])
    print()
    print_kv(
        [
            ("MOSEI BERT GMTM best MAE", mosei_gmtm),
            ("MOSEI BERT baseline best MAE", mosei_base),
            ("GMTM better than best baseline", mosei_gmtm < mosei_base),
        ]
    )
    # Pins the documented headline number so a silent CSV edit fails tests.
    if abs(mosei_gmtm - 0.5640) > 1e-9:
        raise RuntimeError(f"expected GMTM MAE 0.5640, got {mosei_gmtm}")
    if by_title["MOSEI BERT baselines"]["Fusion Method"] != "TransformerLate":
        raise RuntimeError("expected TransformerLate as best BERT baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
