"""Numeric companion to docs/research-notes.md.

Reads the four GMTM ablation CSVs and prints MAE deltas:

* text-only → full (how much non-text helped)
* text-only → text+audio (did audio help or hurt?)
* text-only → text+visual
* audio+visual vs the weaker of audio / visual (does non-text fuse?)
"""

from __future__ import annotations

import csv
from pathlib import Path

import _paths
from _paths import MOSI_TEST_DIR, RESULTS_DIR

ABLATIONS = [
    ("MOSEI BERT", RESULTS_DIR / "ablation_results.csv"),
    ("MOSEI GloVe", RESULTS_DIR / "ablation_glove_results.csv"),
    ("MOSI BERT (pooled)", MOSI_TEST_DIR / "ablation_mosi_results.csv"),
    ("MOSI GloVe (pooled)", MOSI_TEST_DIR / "ablation_mosi_glove_results.csv"),
]


def _normalize_name(raw: str) -> str:
    """``['text', 'audio']`` / ``\"['text', 'audio']\"`` → ``text+audio``."""
    text = raw.strip().strip('"').strip()
    text = text.replace("'", "").replace("[", "").replace("]", "")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return "+".join(parts)


def load_mae(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, float] = {}
    for row in rows:
        name = _normalize_name(row["Fusion Method"])
        out[name] = float(row["MAE"])
    return out


def deltas(mae: dict[str, float]) -> dict[str, float]:
    text = mae["text"]
    full = mae["text+audio+visual"]
    return {
        "text": text,
        "full": full,
        "text→full": text - full,
        "text→text+audio": text - mae["text+audio"],
        "text→text+visual": text - mae["text+visual"],
        "audio": mae["audio"],
        "visual": mae["visual"],
        "audio+visual": mae["audio+visual"],
        "best-unimodal-nontext→av": min(mae["audio"], mae["visual"])
        - mae["audio+visual"],
    }


def _fmt(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.4f}"


def _demo() -> list[dict[str, float]]:
    print("GMTM ablation MAE deltas (positive = improved vs the left side)")
    print(
        f"{'table':<22} {'text':>8} {'full':>8} {'Δ full':>8} "
        f"{'Δ +A':>8} {'Δ +V':>8} {'A+V vs min(A,V)':>16}"
    )
    collected = []
    for title, path in ABLATIONS:
        mae = load_mae(path)
        d = deltas(mae)
        collected.append(d)
        print(
            f"{title:<22} {d['text']:8.4f} {d['full']:8.4f} "
            f"{_fmt(d['text→full']):>8} {_fmt(d['text→text+audio']):>8} "
            f"{_fmt(d['text→text+visual']):>8} "
            f"{_fmt(d['best-unimodal-nontext→av']):>16}"
        )
    print()
    print("sign convention: text→X > 0 means X has lower MAE than text-only.")
    print("MOSEI GloVe text→text+audio is negative in the logged table "
          "(audio hurt).")
    return collected


if __name__ == "__main__":
    _demo()
