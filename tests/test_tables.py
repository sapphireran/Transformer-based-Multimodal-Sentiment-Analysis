"""Sanity-check the recorded CSVs and the compare_tables arithmetic."""

from __future__ import annotations

import csv
from pathlib import Path

from common import MOSI_RESULTS_DIR, RESULTS_DIR
from compare_tables import _by_name, _delta, _f, _load


def test_all_result_csvs_have_six_metrics():
    paths = list(RESULTS_DIR.glob("*.csv")) + [
        MOSI_RESULTS_DIR / "mosi_bert_results.csv",
        MOSI_RESULTS_DIR / "mosi_glove_results.csv",
        MOSI_RESULTS_DIR / "ablation_mosi_results.csv",
        MOSI_RESULTS_DIR / "ablation_mosi_glove_results.csv",
    ]
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        assert rows, path
        for row in rows:
            assert _f(row, "MAE") > 0
            assert 0 <= _f(row, "F1") <= 1
            assert 0 <= _f(row, "ACC2") <= 1


def test_bert_gmtm_full_is_best_mosei_mae():
    abl = _by_name(_load(RESULTS_DIR / "ablation_results.csv"))
    full = _f(abl["['text', 'audio', 'visual']"], "MAE")
    assert full == 0.5640
    assert full == min(_f(row, "MAE") for row in abl.values())


def test_bert_beats_glove_on_every_baseline_mae():
    bert = _by_name(_load(RESULTS_DIR / "main_results.csv"))
    glove = _by_name(_load(RESULTS_DIR / "glove_results.csv"))
    for name in bert:
        assert _delta(bert[name], glove[name], "MAE") < 0, name


def test_glove_text_plus_audio_is_worse_than_text():
    abl = _by_name(_load(RESULTS_DIR / "ablation_glove_results.csv"))
    assert _delta(abl["['text', 'audio']"], abl["['text']"], "MAE") > 0


def test_repo_root_results_copies_match_model_root_copies():
    """The CSVs next to the train scripts should match model/results/."""
    root = Path(__file__).resolve().parents[1] / "model"
    pairs = [
        (root / "main_results.csv", RESULTS_DIR / "main_results.csv"),
        (root / "glove_results.csv", RESULTS_DIR / "glove_results.csv"),
        (root / "ablation_results.csv", RESULTS_DIR / "ablation_results.csv"),
        (root / "ablation_glove_results.csv", RESULTS_DIR / "ablation_glove_results.csv"),
    ]
    for left, right in pairs:
        assert left.read_text() == right.read_text()
