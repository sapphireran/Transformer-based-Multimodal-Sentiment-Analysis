"""Committed CSVs stay readable and keep the numbers the docs quote."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.paths import RESULTS_DIR
from examples.lib.results_tables import (
    format_table,
    load_result_csv,
    mark_best_rows,
    repo_result_tables,
)


def test_all_primary_tables_exist():
    titles = []
    for title, path in repo_result_tables(include_duplicates=False):
        assert path.is_file(), path
        payload = load_result_csv(path)
        assert payload["rows"], path
        titles.append(title)
    assert len(titles) == 8


def test_mosei_bert_bakeoff_late_transformer_wins_mae():
    payload = load_result_csv(RESULTS_DIR / "main_results.csv")
    winners = mark_best_rows(payload["fieldnames"], payload["rows"])
    best_idx = winners["MAE"][1][0]
    assert payload["rows"][best_idx]["Fusion Method"] == "TransformerLate"
    assert winners["MAE"][0] == pytest.approx(0.5846)


def test_mosei_gmtm_all_three_is_best_mae():
    payload = load_result_csv(RESULTS_DIR / "ablation_results.csv")
    winners = mark_best_rows(payload["fieldnames"], payload["rows"])
    best_idx = winners["MAE"][1][0]
    assert "text" in payload["rows"][best_idx]["Fusion Method"]
    assert "audio" in payload["rows"][best_idx]["Fusion Method"]
    assert "visual" in payload["rows"][best_idx]["Fusion Method"]
    assert winners["MAE"][0] == pytest.approx(0.5640)


def test_format_table_marks_winner():
    payload = load_result_csv(RESULTS_DIR / "main_results.csv")
    text = format_table("demo", payload)
    assert "TransformerLate" in text
    assert "best:" in text
    assert "*0.5846" in text or "*0.5846" in text.replace(" ", "")
