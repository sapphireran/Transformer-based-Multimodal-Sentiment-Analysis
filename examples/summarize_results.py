"""Reprint the committed CSVs as markdown tables.

This does not retrain anything. It is the same transcription
``docs/results.md`` started from, so a drifted CSV is visible.

    python examples/summarize_results.py
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pandas as pd

from common import MOSI_RESULTS_DIR, REPO_ROOT, RESULTS_DIR

OUT_DIR = REPO_ROOT / "examples" / "output"
METRIC_COLS = ["MAE", "ACC7", "Acc5", "ACC5", "ACC2", "Corr", "F1"]


def _normalize_method(value) -> str:
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return "+".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            text = text.strip("[]").replace("'", "").replace('"', "")
            return "+".join(part.strip() for part in text.split(",") if part.strip())
    return text


def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "Fusion Method"]


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Fusion Method" not in df.columns:
        raise ValueError(f"{path} has no 'Fusion Method' column: {list(df.columns)}")
    df = df.copy()
    df["Fusion Method"] = df["Fusion Method"].map(_normalize_method)
    return df


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["Fusion Method", *_pick_metric_cols(df)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" if c == "Fusion Method" else "---:" for c in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [str(row["Fusion Method"])]
        for c in cols[1:]:
            val = row[c]
            cells.append(f"{float(val):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLES = [
    ("MOSEI BERT fusion sweep", RESULTS_DIR / "main_results.csv"),
    ("MOSEI BERT GMTM ablation", RESULTS_DIR / "ablation_results.csv"),
    ("MOSEI GloVe fusion sweep", RESULTS_DIR / "glove_results.csv"),
    ("MOSEI GloVe GMTM ablation", RESULTS_DIR / "ablation_glove_results.csv"),
    ("MOSI transfer BERT", MOSI_RESULTS_DIR / "mosi_bert_results.csv"),
    ("MOSI transfer BERT GMTM ablation", MOSI_RESULTS_DIR / "ablation_mosi_results.csv"),
    ("MOSI transfer GloVe", MOSI_RESULTS_DIR / "mosi_glove_results.csv"),
    ("MOSI transfer GloVe GMTM ablation", MOSI_RESULTS_DIR / "ablation_mosi_glove_results.csv"),
]


def run(write: bool = True) -> str:
    chunks = ["# Result tables (from committed CSVs)", ""]
    for title, path in TABLES:
        if not path.is_file():
            raise FileNotFoundError(path)
        df = load_table(path)
        chunks.append(f"## {title}")
        chunks.append("")
        chunks.append(f"Source: `{path.relative_to(REPO_ROOT)}`")
        chunks.append("")
        chunks.append(to_markdown(df))
        chunks.append("")
        best = df.loc[df["MAE"].idxmin(), "Fusion Method"]
        chunks.append(f"Lowest MAE: **{best}** ({df['MAE'].min():.4f}).")
        chunks.append("")
        print(f"{title}: {len(df)} rows, best MAE = {best}")

    text = "\n".join(chunks).rstrip() + "\n"
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "result_tables.md"
        out.write_text(text, encoding="utf-8")
        print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return text


if __name__ == "__main__":
    run()
