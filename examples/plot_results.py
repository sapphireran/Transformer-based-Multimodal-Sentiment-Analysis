#!/usr/bin/env python3
"""Headless charts for the committed MOSI / MOSEI CSVs.

Replaces the interactive cells in ``model/results/plot.ipynb``: append the
full GMTM row onto each fusion-zoo table, then write BERT vs GloVe line
and bar figures under ``examples/output/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_DIR, ensure_output_dir  # noqa: E402
from inspect_results import load_table  # noqa: E402

METRICS: Tuple[str, ...] = ("MAE", "Acc7", "Acc5", "Acc2", "Corr", "F1")
ALIASES = {
    "MAE": ("MAE",),
    "Acc7": ("Acc7", "ACC7", "Acc7_uniform"),
    "Acc5": ("Acc5", "ACC5", "Acc5_uniform"),
    "Acc2": ("Acc2", "ACC2"),
    "Corr": ("Corr",),
    "F1": ("F1",),
}


def metric_value(row: Dict[str, str], name: str) -> float:
    for alias in ALIASES[name]:
        if alias in row:
            return float(row[alias])
    raise KeyError(f"{name} not in row keys {sorted(row)}")


def with_gmtm(fusion_rows: List[Dict[str, str]], ablation_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append the last ablation row as ``GatedMultiTransformer``, matching the notebook."""
    merged = [dict(row) for row in fusion_rows]
    gmtm = dict(ablation_rows[-1])
    gmtm["Fusion Method"] = "GatedMultiTransformer"
    merged.append(gmtm)
    return merged


def _series(rows: Sequence[Dict[str, str]], metric: str) -> np.ndarray:
    return np.asarray([metric_value(row, metric) for row in rows], dtype=np.float64)


def plot_fusion_and_ablation(out_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_rows = load_table(MODEL_DIR / "main_results.csv")
    glove_rows = load_table(MODEL_DIR / "glove_results.csv")
    abl_bert = load_table(MODEL_DIR / "ablation_results.csv")
    abl_glove = load_table(MODEL_DIR / "ablation_glove_results.csv")

    fusion_bert = with_gmtm(main_rows, abl_bert)
    fusion_glove = with_gmtm(glove_rows, abl_glove)
    fusion_names = [row["Fusion Method"] for row in fusion_bert]
    abl_names = [row["Fusion Method"] for row in abl_bert]

    written: List[Path] = []

    def _line_figure(names: Sequence[str], left_rows, right_rows, title: str, filename: str) -> Path:
        x = np.arange(len(names))
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(title)
        for i, metric in enumerate(METRICS):
            ax = axes[i // 3, i % 3]
            ax.plot(x, _series(left_rows, metric), marker="o", label="BERT")
            ax.plot(x, _series(right_rows, metric), marker="s", linestyle="--", label="GloVe")
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("lower better" if metric == "MAE" else "higher better")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _bar_figure(names: Sequence[str], left_rows, right_rows, title: str, filename: str) -> Path:
        x = np.arange(len(names))
        width = 0.38
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(title)
        for i, metric in enumerate(METRICS):
            ax = axes[i // 3, i % 3]
            left = _series(left_rows, metric)
            right = _series(right_rows, metric)
            ax.bar(x - width / 2, left, width=width, label="BERT")
            ax.bar(x + width / 2, right, width=width, label="GloVe")
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("lower better" if metric == "MAE" else "higher better")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(fontsize=8)
            ax.set_ylim(0, max(left.max(), right.max()) * 1.25)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    written.append(
        _line_figure(
            fusion_names,
            fusion_bert,
            fusion_glove,
            "MOSEI fusion zoo (GMTM appended from ablation last row)",
            "mosei_fusion_zoo_lines.png",
        )
    )
    written.append(
        _line_figure(
            abl_names,
            abl_bert,
            abl_glove,
            "MOSEI GMTM modality ablation",
            "mosei_ablation_lines.png",
        )
    )
    written.append(
        _bar_figure(
            abl_names,
            abl_bert,
            abl_glove,
            "MOSEI GMTM modality ablation (bars)",
            "mosei_ablation_bars.png",
        )
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="", help="defaults to examples/output/")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else ensure_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_fusion_and_ablation(out_dir)
    for path in paths:
        print(f"wrote {path}  ({path.stat().st_size} bytes)")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
