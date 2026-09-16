"""Pretty-print the checked-in personal result CSVs and rank each column.

This is the executable twin of docs/results.md: if a CSV changes, rerun
this script instead of hand-editing ranks. Paths are relative to the
repository root.
"""

from __future__ import annotations

import csv
from pathlib import Path

from common import REPO_ROOT

TABLES = [
    ("MOSEI BERT fusion", REPO_ROOT / "model" / "results" / "main_results.csv"),
    ("MOSEI GloVe fusion", REPO_ROOT / "model" / "results" / "glove_results.csv"),
    ("MOSEI BERT GMTM ablation", REPO_ROOT / "model" / "results" / "ablation_results.csv"),
    ("MOSEI GloVe GMTM ablation", REPO_ROOT / "model" / "results" / "ablation_glove_results.csv"),
    ("MOSI BERT transfer (merged splits)", REPO_ROOT / "model" / "mosi_test" / "mosi_bert_results.csv"),
    ("MOSI GloVe transfer (merged splits)", REPO_ROOT / "model" / "mosi_test" / "mosi_glove_results.csv"),
    ("MOSI BERT GMTM ablation", REPO_ROOT / "model" / "mosi_test" / "ablation_mosi_results.csv"),
    ("MOSI GloVe GMTM ablation", REPO_ROOT / "model" / "mosi_test" / "ablation_mosi_glove_results.csv"),
]

# Lower is better only for MAE. Everything else is higher-is-better.
MINIMIZE = {"MAE", "mae"}


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0], rows[1:]


def _as_float(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def _best_index(values: list[float | None], minimize: bool) -> int | None:
    ranked = [(i, v) for i, v in enumerate(values) if v is not None]
    if not ranked:
        return None
    key = (lambda iv: iv[1]) if minimize else (lambda iv: -iv[1])
    return min(ranked, key=key)[0]


def format_table(title: str, path: Path) -> str:
    header, body = _read(path)
    numeric_cols = []
    for j, name in enumerate(header):
        if j == 0:
            continue
        column = [_as_float(row[j]) if j < len(row) else None for row in body]
        if any(v is not None for v in column):
            numeric_cols.append((j, name, column))

    winners = {}
    for j, name, column in numeric_cols:
        winners[j] = _best_index(column, name.strip() in MINIMIZE or name.lower() == "mae")

    widths = [max(len(header[j]), *(len(row[j]) if j < len(row) else 0 for row in body)) for j in range(len(header))]
    # Leave room for a trailing *
    widths = [w + 1 for w in widths]

    def fmt_row(cells: list[str], star_at: set[int] | None = None) -> str:
        parts = []
        for j, cell in enumerate(cells):
            text = cell
            if star_at and j in star_at:
                text = cell + "*"
            parts.append(text.ljust(widths[j] if j < len(widths) else len(text) + 1))
        return "  ".join(parts)

    lines = [title, str(path.relative_to(REPO_ROOT)), fmt_row(header), "-" * 72]
    for i, row in enumerate(body):
        stars = {j for j, win in winners.items() if win == i}
        lines.append(fmt_row(row, stars))
    lines.append("(* best in column; MAE minimized, other scores maximized)")
    return "\n".join(lines)


def summarize_takeaways() -> str:
    """One-screen reminder of the docs/results.md narrative."""
    return "\n".join(
        [
            "Takeaways from the starred cells:",
            "  • MOSEI BERT fusion: TransformerLate wins every metric.",
            "  • MOSEI GloVe fusion: LowRankTensorFusion wins every metric.",
            "  • MOSEI BERT GMTM: full text+audio+visual is best on MAE/Acc2/Corr/F1.",
            "  • MOSI BERT transfer: TransformerLate has the best MAE among fusion rows.",
            "  • MOSI GloVe transfer: GMTM wins every metric.",
            "  • MOSI loaders merge train+valid+test; do not quote as official test Acc.",
        ]
    )


def main() -> None:
    for title, path in TABLES:
        if not path.exists():
            print(f"MISSING {path}")
            print()
            continue
        print(format_table(title, path))
        print()
    print(summarize_takeaways())


if __name__ == "__main__":
    main()
