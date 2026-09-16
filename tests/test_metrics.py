"""Unit tests for the documented MOSI/MOSEI metric helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.metrics import (
    binary_acc_f1,
    regression_metrics,
    split_uniform_5,
    split_uniform_7,
    summarize_predictions,
    uniform_edges,
)
from examples.metrics_walkthrough import Y, YHAT


def test_uniform_7_edges_match_training_formula():
    edges = uniform_edges(7)
    step = 6.0 / 7.0
    expected = tuple(-3.0 + i * step for i in range(8))
    assert edges == expected
    assert edges[0] == pytest.approx(-3.0)
    assert edges[-1] == pytest.approx(3.0)


def test_uniform_5_edges_match_training_formula():
    edges = uniform_edges(5)
    step = 6.0 / 5.0
    expected = tuple(-3.0 + i * step for i in range(6))
    assert edges == expected


def test_digitize_endpoints():
    # x == -3 lands in bin 1; x == 3 is clipped to the last bin.
    assert split_uniform_7([-3.0, 3.0]).tolist() == [1, 7]
    assert split_uniform_5([-3.0, 3.0]).tolist() == [1, 5]


def test_midpoints_land_in_expected_7_bins():
    edges = uniform_edges(7)
    mids = [(edges[i] + edges[i + 1]) / 2.0 for i in range(7)]
    assert split_uniform_7(mids).tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_exclude_zero_drops_neutrals():
    y = np.array([-1.0, 0.0, 0.0, 2.0])
    yhat = np.array([-0.5, 1.0, -1.0, 1.5])
    f1, acc = binary_acc_f1(y, yhat, exclude_zero=True)
    # only the two non-zero gold labels remain; both signs match
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_exclude_zero_false_keeps_neutrals_as_negative():
    y = np.array([0.0, 1.0])
    yhat = np.array([0.2, 0.8])  # 0.2 > 0 so gold-neutral is a false positive
    f1, acc = binary_acc_f1(y, yhat, exclude_zero=False)
    assert acc == pytest.approx(0.5)


def test_perfect_regression():
    y = np.linspace(-3, 3, 21)
    out = regression_metrics(y, y)
    assert out["MAE"] == pytest.approx(0.0)
    assert out["MSE"] == pytest.approx(0.0)
    assert out["Corr"] == pytest.approx(1.0)


def test_constant_pred_corr_is_nan():
    y = np.array([-1.0, 0.0, 2.0])
    out = regression_metrics(y, np.ones_like(y))
    assert math.isnan(out["Corr"])


def test_walkthrough_vectors_are_stable():
    summary = summarize_predictions(Y, YHAT)
    # Locked so a later rewrite of the walkthrough cannot silently change docs.
    assert summary["MAE"] == pytest.approx(0.375)
    assert summary["MSE"] == pytest.approx(0.1758333333)
    assert summary["Corr"] == pytest.approx(0.9779185402)
    assert summary["Acc7_uniform"] == pytest.approx(2 / 3)
    assert summary["Acc5_uniform"] == pytest.approx(5 / 6)
    assert summary["Acc2"] == pytest.approx(0.9)
    assert summary["F1"] == pytest.approx(0.9230769231)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_predictions([1.0, 2.0], [1.0])
