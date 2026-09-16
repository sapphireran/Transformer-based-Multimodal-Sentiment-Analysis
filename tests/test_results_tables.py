"""The eight experiment CSVs used by docs/examples must stay parseable."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"

CSVS = [
    MODEL_DIR / "results" / "main_results.csv",
    MODEL_DIR / "results" / "ablation_results.csv",
    MODEL_DIR / "results" / "glove_results.csv",
    MODEL_DIR / "results" / "ablation_glove_results.csv",
    MODEL_DIR / "mosi_test" / "mosi_bert_results.csv",
    MODEL_DIR / "mosi_test" / "ablation_mosi_results.csv",
    MODEL_DIR / "mosi_test" / "mosi_glove_results.csv",
    MODEL_DIR / "mosi_test" / "ablation_mosi_glove_results.csv",
]


def test_all_result_csvs_exist_and_have_metric_columns():
    required = {"mae", "corr", "f1"}
    for path in CSVS:
        assert path.exists(), path
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) >= 2, path
        header = [c.strip().lower() for c in rows[0]]
        joined = " ".join(header)
        for needle in required:
            assert needle in joined, f"{path} missing {needle} (header={rows[0]})"
        for row in rows[1:]:
            assert len(row) == len(rows[0]), path
