"""Tests for the dataset-free sentiment metrics."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    CSV_COLUMNS,
    eval_affect,
    evaluate_sentiment,
    regression_scores,
    row_for_csv,
    split_uniform_5,
    split_uniform_7,
    summarize_bin_edges,
)


def test_uniform_7_endpoints():
    labels = split_uniform_7([-3.0, 0.0, 3.0])
    assert labels.tolist() == [1, 4, 7]


def test_uniform_5_monotonic():
    grid = np.linspace(-3, 3, 11)
    labels = split_uniform_5(grid)
    assert labels.min() == 1
    assert labels.max() == 5
    assert np.all(np.diff(labels) >= 0)


def test_bin_edges_cover_interval():
    edges = summarize_bin_edges(7)
    assert edges[0][1] == -3.0
    assert abs(edges[-1][2] - 3.0) < 1e-9
    assert len(edges) == 7


def test_eval_affect_drops_neutral():
    truths = np.array([-2.0, 0.0, 1.5, 0.0, 2.0])
    preds = np.array([-1.0, 4.0, 1.0, -4.0, 0.2])
    f1, acc = eval_affect(truths, preds, exclude_zero=True)
    # three kept samples: -, +, + all correct
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_eval_affect_keeps_neutral_when_asked():
    truths = np.array([0.0, 1.0])
    preds = np.array([0.5, 1.0])  # 0.5 > 0 so neutral truth is "wrong"
    f1, acc = eval_affect(truths, preds, exclude_zero=False)
    assert acc == pytest.approx(0.5)


def test_oracle_metrics_are_perfect():
    y = np.array([-2.5, -0.2, 0.4, 1.7, 2.9])
    scores = evaluate_sentiment(y, y)
    assert scores["MAE"] == pytest.approx(0.0)
    assert scores["MSE"] == pytest.approx(0.0)
    assert scores["Corr"] == pytest.approx(1.0)
    assert scores["Acc7_uniform"] == pytest.approx(1.0)
    assert scores["Acc5_uniform"] == pytest.approx(1.0)
    assert scores["Acc2"] == pytest.approx(1.0)
    assert scores["F1"] == pytest.approx(1.0)


def test_sign_flip_hurts_correlation():
    y = np.linspace(-2.5, 2.5, 20)
    flipped = regression_scores(y, -y)
    assert flipped["Corr"] == pytest.approx(-1.0)
    assert flipped["MAE"] > 1.0


def test_csv_row_header_alignment():
    scores = evaluate_sentiment([1.0, -1.0], [1.0, -1.0])
    row = row_for_csv("ConcatLate", scores)
    assert len(row) == len(CSV_COLUMNS)
    assert row[0] == "ConcatLate"
    assert row[1] == 0.0
