"""Unit tests for the stand-alone MOSI/MOSEI metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    as_1d_numpy,
    classification_scores,
    eval_affect,
    evaluate_sentiment,
    format_score_table,
    regression_scores,
    split_uniform,
    split_uniform_5,
    split_uniform_7,
)


def test_as_1d_numpy_squeezes_column_vector():
    assert as_1d_numpy([[1.0], [2.0]]).tolist() == [1.0, 2.0]


def test_eval_affect_perfect_nonzero_protocol():
    truths = np.array([-2.0, -0.1, 0.0, 0.4, 2.5])
    preds = np.array([-1.0, -0.2, 0.3, 1.1, 2.0])
    f1, acc = eval_affect(truths, preds, exclude_zero=True)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_eval_affect_keeps_zero_when_asked():
    truths = np.array([0.0, 1.0])
    preds = np.array([-0.1, 0.5])
    # zero is treated as negative when exclude_zero is False
    f1, acc = eval_affect(truths, preds, exclude_zero=False)
    assert acc == pytest.approx(1.0)
    assert 0.0 <= f1 <= 1.0


def test_eval_affect_empty_after_drop():
    f1, acc = eval_affect([0.0, 0.0], [1.0, -1.0], exclude_zero=True)
    assert (f1, acc) == (0.0, 0.0)


def test_uniform_bins_cover_endpoints():
    # -3 → first bin (1), +3 → last bin
    assert split_uniform_7([-3.0])[0] == 1
    assert split_uniform_7([3.0])[0] == 7
    assert split_uniform_5([-3.0])[0] == 1
    assert split_uniform_5([3.0])[0] == 5


def test_split_uniform_rejects_tiny_bin_count():
    with pytest.raises(ValueError):
        split_uniform([0.0], n_bins=1)


def test_regression_scores_perfect():
    y = np.array([-1.5, 0.0, 2.25])
    scores = regression_scores(y, y)
    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["MSE"] == pytest.approx(0.0)
    assert scores["Corr"] == pytest.approx(1.0)


def test_regression_scores_constant_pred_has_zero_corr():
    y = np.array([-1.0, 0.0, 2.0])
    scores = regression_scores(y, np.zeros_like(y))
    assert scores["Corr"] == 0.0
    assert scores["MAE"] == pytest.approx(np.mean(np.abs(y)))


def test_classification_and_bundle_keys():
    y = np.array([-2.0, -0.5, 0.2, 1.8])
    pred = y + 0.05
    class_scores = classification_scores(y, pred)
    bundle = evaluate_sentiment(y, pred)
    for key in ("Acc7_uniform", "Acc5_uniform", "Acc2", "F1"):
        assert key in class_scores
        assert key in bundle
    for key in ("MAE", "MSE", "Corr"):
        assert key in bundle


def test_format_score_table_contains_method_name():
    table = format_score_table({"demo": {"MAE": 0.5, "Acc2": 1.0}})
    assert "demo" in table
    assert "0.5000" in table
    assert format_score_table({}) == "_no scores_"
