#!/usr/bin/env python3
"""Redraw the published CSVs as comparison bar charts (headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ensure_output_dir
from results_lib import ResultRow, published_tables


def _bar(ax, rows: List[ResultRow], value, ylabel: str, title: str, lower_is_better: bool) -> None:
    labels = [r.method for r in rows]
    values = [value(r) for r in rows]
    colors = []
    best = min(values) if lower_is_better else max(values)
    for v in values:
        colors.append("#2a6f97" if v == best else "#89b0c4")
    ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> int:
    tables = published_tables()
    out = ensure_output_dir()
    paths: List[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _bar(
        axes[0],
        tables["mosei-bert-fusion"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSEI + BERT fusion",
        True,
    )
    _bar(
        axes[1],
        tables["mosei-glove-fusion"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSEI + GloVe fusion",
        True,
    )
    paths.append(_save(fig, out / "mosei_mae_comparison.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    _bar(
        axes[0],
        tables["mosei-bert-ablation"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSEI BERT GMTM ablation",
        True,
    )
    _bar(
        axes[1],
        tables["mosei-glove-ablation"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSEI GloVe GMTM ablation",
        True,
    )
    paths.append(_save(fig, out / "mosei_ablation_mae.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    _bar(
        axes[0],
        tables["mosi-bert-fusion"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSI transfer + BERT",
        True,
    )
    _bar(
        axes[1],
        tables["mosi-glove-fusion"],
        lambda r: r.mae,
        "MAE (lower better)",
        "MOSI transfer + GloVe",
        True,
    )
    paths.append(_save(fig, out / "mosi_transfer_mae.png"))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _bar(
        ax,
        tables["mosei-bert-fusion"],
        lambda r: r.f1,
        "F1 (higher better)",
        "MOSEI + BERT  binary F1",
        False,
    )
    paths.append(_save(fig, out / "mosei_bert_f1.png"))

    manifest = out / "plot_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
