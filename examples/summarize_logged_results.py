"""Pretty-print every checked-in result CSV and mark per-column winners.

Reads the personal logs under ``model/results/`` and
``model/mosi_test/``. No model weights required.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _paths
from _paths import MOSI_TEST_DIR, RESULTS_DIR

TABLES = [
    ("MOSEI BERT baselines", RESULTS_DIR / "main_results.csv", "min-mae"),
    ("MOSEI BERT GMTM ablation", RESULTS_DIR / "ablation_results.csv", "min-mae"),
    ("MOSEI GloVe baselines", RESULTS_DIR / "glove_results.csv", "min-mae"),
    ("MOSEI GloVe GMTM ablation", RESULTS_DIR / "ablation_glove_results.csv", "min-mae"),
    ("MOSI BERT transfer (pooled)", MOSI_TEST_DIR / "mosi_bert_results.csv", "min-mae"),
    ("MOSI BERT GMTM ablation", MOSI_TEST_DIR / "ablation_mosi_results.csv", "min-mae"),
    ("MOSI GloVe transfer (pooled)", MOSI_TEST_DIR / "mosi_glove_results.csv", "min-mae"),
    ("MOSI GloVe GMTM ablation", MOSI_TEST_DIR / "ablation_mosi_glove_results.csv", "min-mae"),
]

LOWER_IS_BETTER = {"mae"}


def _load(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _winners(header: list[str], body: list[list[str]]) -> dict[int, set[int]]:
    """column index → set of winning row indices."""
    marks: dict[int, set[int]] = {}
    if not body:
        return marks
    n_cols = max(len(header), max(len(r) for r in body))
    for col in range(1, n_cols):
        name = header[col].strip().lower() if col < len(header) else ""
        parsed: list[tuple[int, float]] = []
        for i, row in enumerate(body):
            if col >= len(row):
                continue
            number = _as_float(row[col])
            if number is not None:
                parsed.append((i, number))
        if not parsed:
            continue
        if name in LOWER_IS_BETTER:
            best = min(v for _, v in parsed)
        else:
            best = max(v for _, v in parsed)
        marks[col] = {i for i, v in parsed if abs(v - best) < 1e-12}
    return marks


def format_table(title: str, path: Path) -> str:
    if not path.is_file():
        return f"## {title}\nmissing: {path}\n"
    header, body = _load(path)
    marks = _winners(header, body)
    widths = [len(h) for h in header]
    display_rows: list[list[str]] = []
    for i, row in enumerate(body):
        cells = []
        for c, cell in enumerate(row):
            text = cell
            if c in marks and i in marks[c]:
                text = f"*{cell}*"
            cells.append(text)
            if c >= len(widths):
                widths.append(len(text))
            else:
                widths[c] = max(widths[c], len(text))
        display_rows.append(cells)

    def fmt(cells: list[str]) -> str:
        parts = []
        for c, cell in enumerate(cells):
            width = widths[c] if c < len(widths) else len(cell)
            parts.append(cell.ljust(width) if c == 0 else cell.rjust(width))
        return " | ".join(parts)

    lines = [
        f"## {title}",
        f"source: {path.relative_to(_paths.REPO_ROOT)}",
        fmt(header),
        " | ".join("-" * w for w in widths),
    ]
    lines.extend(fmt(row) for row in display_rows)
    lines.append("(*best in column: min MAE, max otherwise*)")
    return "\n".join(lines) + "\n"


def text_dominance_note() -> str:
    """One quantitative reminder from the BERT MOSEI ablation CSV."""
    path = RESULTS_DIR / "ablation_results.csv"
    if not path.is_file():
        return ""
    header, body = _load(path)
    try:
        mae_i = next(i for i, name in enumerate(header) if name.lower() == "mae")
    except StopIteration:
        return ""
    by_name = {row[0]: float(row[mae_i]) for row in body}
    text = by_name.get("['text']")
    full = by_name.get("['text', 'audio', 'visual']")
    audio = by_name.get("['audio']")
    if None in (text, full, audio):
        return ""
    return (
        "BERT MOSEI GMTM: text-only MAE "
        f"{text:.4f} vs full {full:.4f} (Δ {text - full:.4f}); "
        f"audio-only {audio:.4f}."
    )


def _demo() -> list[str]:
    blocks = [format_table(title, path) for title, path, _ in TABLES]
    note = text_dominance_note()
    if note:
        blocks.append(note)
    text = "\n".join(blocks)
    print(text)
    return blocks


if __name__ == "__main__":
    _demo()
