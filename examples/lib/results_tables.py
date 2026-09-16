"""Load the committed experiment CSVs and mark best cells."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .paths import MODEL_DIR, MOSI_TEST_DIR, RESULTS_DIR

# Lower is better / higher is better. Column names vary slightly across files.
MINIMIZE = {"mae", "testloss", "mse"}
MAXIMIZE = {"acc7", "acc5", "acc2", "corr", "f1"}

REPO_TABLES: Tuple[Tuple[str, Path], ...] = (
    ("MOSEI BERT bake-off", RESULTS_DIR / "main_results.csv"),
    ("MOSEI GloVe bake-off", RESULTS_DIR / "glove_results.csv"),
    ("MOSEI GMTM BERT ablation", RESULTS_DIR / "ablation_results.csv"),
    ("MOSEI GMTM GloVe ablation", RESULTS_DIR / "ablation_glove_results.csv"),
    ("MOSI transfer BERT bake-off", MOSI_TEST_DIR / "mosi_bert_results.csv"),
    ("MOSI transfer GloVe bake-off", MOSI_TEST_DIR / "mosi_glove_results.csv"),
    ("MOSI GMTM BERT ablation", MOSI_TEST_DIR / "ablation_mosi_results.csv"),
    ("MOSI GMTM GloVe ablation", MOSI_TEST_DIR / "ablation_mosi_glove_results.csv"),
    # duplicates at model/ root (same numbers, kept in the original upload)
    ("MOSEI BERT bake-off (model/ copy)", MODEL_DIR / "main_results.csv"),
    ("MOSEI GloVe bake-off (model/ copy)", MODEL_DIR / "glove_results.csv"),
)


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_result_csv(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"no header in {path}")
        rows = [dict(row) for row in reader]
    return {"path": path, "fieldnames": list(reader.fieldnames), "rows": rows}


def _metric_kind(header: str) -> str:
    key = _norm_header(header)
    if key in MINIMIZE or key.endswith("mae") or key.endswith("mse"):
        return "min"
    if key in MAXIMIZE or key.startswith("acc") or key in {"corr", "f1"}:
        return "max"
    return "skip"


def _float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def mark_best_rows(fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Return per-metric winners (column → (best_value, row_indices))."""
    winners: Dict[str, Tuple[float, List[int]]] = {}
    for col in fieldnames[1:]:
        kind = _metric_kind(col)
        if kind == "skip":
            continue
        values = [_float_or_none(row.get(col, "")) for row in rows]
        numbered = [(i, v) for i, v in enumerate(values) if v is not None]
        if not numbered:
            continue
        target = min((v for _, v in numbered)) if kind == "min" else max((v for _, v in numbered))
        idxs = [i for i, v in numbered if v == target]
        winners[col] = (target, idxs)
    return winners


def format_table(title: str, payload: Dict[str, object], highlight: bool = True) -> str:
    fieldnames: List[str] = payload["fieldnames"]  # type: ignore[assignment]
    rows: List[Dict[str, str]] = payload["rows"]  # type: ignore[assignment]
    winners = mark_best_rows(fieldnames, rows) if highlight else {}
    widths = [max(len(col), *(len(str(row.get(col, ""))) + 4 for row in rows)) for col in fieldnames]
    widths = [max(w, len(col) + 2) for w, col in zip(widths, fieldnames)]

    def cell(col: str, raw: str, row_idx: int | None) -> str:
        text = str(raw)
        if (
            highlight
            and row_idx is not None
            and col in winners
            and row_idx in winners[col][1]
        ):
            text = f"*{text}"
        return text

    header = "  ".join(col.ljust(w) for col, w in zip(fieldnames, widths))
    lines = [title, str(payload["path"]), header, "-" * len(header)]
    for i, row in enumerate(rows):
        line = "  ".join(
            cell(col, row.get(col, ""), i).ljust(w) for col, w in zip(fieldnames, widths)
        )
        lines.append(line)
    if winners:
        bits = []
        for col, (value, idxs) in winners.items():
            names = [rows[i][fieldnames[0]] for i in idxs]
            bits.append(f"{col}={value:g} ({', '.join(names)})")
        lines.append("best: " + "; ".join(bits))
    return "\n".join(lines)


def repo_result_tables(include_duplicates: bool = False) -> Iterable[Tuple[str, Path]]:
    seen_paths = set()
    for title, path in REPO_TABLES:
        if not include_duplicates and path.parent == MODEL_DIR and path.name in {
            "main_results.csv",
            "glove_results.csv",
            "ablation_results.csv",
            "ablation_glove_results.csv",
        }:
            # prefer model/results/ copies
            if (RESULTS_DIR / path.name).is_file():
                continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        yield title, path
