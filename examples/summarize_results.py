#!/usr/bin/env python3
"""Rank the recorded experiment CSVs that already live in this repo.

No pickles or weights required. Default paths are the copies under
``model/results/`` and ``model/mosi_test/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from synthetic_data import REPO_ROOT

METRIC_KEYS = ("MAE", "ACC7", "ACC5", "ACC2", "Corr", "F1")


TABLES = {
    "mosei_bert_fusion": REPO_ROOT / "model" / "results" / "main_results.csv",
    "mosei_glove_fusion": REPO_ROOT / "model" / "results" / "glove_results.csv",
    "mosei_bert_gmtm_ablation": REPO_ROOT / "model" / "results" / "ablation_results.csv",
    "mosei_glove_gmtm_ablation": REPO_ROOT / "model" / "results" / "ablation_glove_results.csv",
    "mosi_bert_transfer": REPO_ROOT / "model" / "mosi_test" / "mosi_bert_results.csv",
    "mosi_glove_transfer": REPO_ROOT / "model" / "mosi_test" / "mosi_glove_results.csv",
    "mosi_bert_gmtm_ablation": REPO_ROOT / "model" / "mosi_test" / "ablation_mosi_results.csv",
    "mosi_glove_gmtm_ablation": REPO_ROOT / "model" / "mosi_test" / "ablation_mosi_glove_results.csv",
}


def _norm_header(name: str) -> str:
    key = name.strip()
    aliases = {
        "Fusion Method": "name",
        "ACC7": "ACC7",
        "Acc7": "ACC7",
        "ACC5": "ACC5",
        "Acc5": "ACC5",
        "ACC2": "ACC2",
        "Acc2": "ACC2",
        "Corr": "Corr",
        "F1": "F1",
        "MAE": "MAE",
    }
    return aliases.get(key, key)


def load_table(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            item: Dict[str, object] = {}
            for key, value in raw.items():
                norm = _norm_header(key)
                if norm in METRIC_KEYS:
                    item[norm] = float(value)
                elif norm == "name":
                    item["name"] = value
                else:
                    item[norm] = value
            rows.append(item)
    return rows


def summarize_table(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_mae = sorted(rows, key=lambda r: float(r["MAE"]))
    by_acc7 = sorted(rows, key=lambda r: float(r["ACC7"]), reverse=True)
    return {
        "n": len(rows),
        "best_mae": {"name": by_mae[0]["name"], "MAE": by_mae[0]["MAE"]},
        "best_acc7": {"name": by_acc7[0]["name"], "ACC7": by_acc7[0]["ACC7"]},
        "ranking_mae": [
            {"rank": i + 1, "name": row["name"], "MAE": row["MAE"], "ACC7": row["ACC7"]}
            for i, row in enumerate(by_mae)
        ],
    }


def run() -> dict:
    report = {}
    for name, path in TABLES.items():
        if not path.exists():
            report[name] = {"missing": str(path)}
            continue
        rows = load_table(path)
        report[name] = summarize_table(rows)
        report[name]["path"] = str(path.relative_to(REPO_ROOT))
    return report


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
