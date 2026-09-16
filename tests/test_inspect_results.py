"""The checked-in result CSVs still parse and have the expected columns."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {"MAE"}
KNOWN = [
    ROOT / "model" / "results" / "main_results.csv",
    ROOT / "model" / "ablation_results.csv",
    ROOT / "model" / "results" / "glove_results.csv",
    ROOT / "model" / "ablation_glove_results.csv",
    ROOT / "model" / "mosi_test" / "mosi_bert_results.csv",
    ROOT / "model" / "mosi_test" / "ablation_mosi_results.csv",
    ROOT / "model" / "mosi_test" / "mosi_glove_results.csv",
    ROOT / "model" / "mosi_test" / "ablation_mosi_glove_results.csv",
]


def test_every_published_csv_exists_and_has_rows():
    for path in KNOWN:
        assert path.is_file(), path
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, f"{path} is empty"
        missing = REQUIRED - set(rows[0])
        assert not missing, f"{path} missing {missing}"
        for row in rows:
            mae = float(row["MAE"])
            assert mae >= 0.0
