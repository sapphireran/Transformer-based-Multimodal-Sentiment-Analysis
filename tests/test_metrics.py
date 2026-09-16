"""Unit tests for the shared MOSI / MOSEI metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model"))

from metrics import (  # noqa: E402
    compute_affect_metrics,
    eval_affect,
    format_metrics_row,
    regression_metrics,
    split_round_7,
    split_uniform_5,
    split_uniform_7,
)


def test_perfect_copy_is_exact():
    y = np.array([-2.4, -1.1, 0.6, 1.8, 2.7])
    metrics = compute_affect_metrics(y, y)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["MSE"] == pytest.approx(0.0)
    assert metrics["Corr"] == pytest.approx(1.0)
    assert metrics["Acc7_uniform"] == pytest.approx(1.0)
    assert metrics["Acc5_uniform"] == pytest.approx(1.0)
    assert metrics["Acc2"] == pytest.approx(1.0)
    assert metrics["F1"] == pytest.approx(1.0)


def test_sign_flip_keeps_mae_but_kills_acc2():
    y = np.array([-2.4, -1.1, 0.6, 1.8, 2.7])
    metrics = compute_affect_metrics(y, -y)
    assert metrics["MAE"] == pytest.approx(np.mean(np.abs(2 * y)))
    assert metrics["Corr"] == pytest.approx(-1.0)
    assert metrics["Acc2"] == pytest.approx(0.0)


def test_exclude_zero_drops_neutrals():
    truth = np.array([-1.0, 0.0, 0.0, 2.0])
    pred = np.array([-0.5, 1.0, -1.0, 1.5])
    f1, acc = eval_affect(truth, pred, exclude_zero=True)
    # Only the two non-zero gold labels; both signs match.
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    f1_all, acc_all = eval_affect(truth, pred, exclude_zero=False)
    # Neutrals (y == 0) are treated as the negative class; preds are + and -.
    assert acc_all == pytest.approx(0.75)


def test_exclude_zero_all_neutral():
    f1, acc = eval_affect(np.zeros(4), np.ones(4), exclude_zero=True)
    assert f1 == 0.0
    assert acc == 0.0


def test_uniform_7_edges():
    # digitize is left-closed / right-open on these edges, then clipped.
    values = np.array([-3.0, -3.0 + 6.0 / 7.0 - 1e-9, 0.0, 3.0])
    bins = split_uniform_7(values)
    assert bins[0] == 1
    assert bins[1] == 1
    assert bins[-1] == 7
    assert 1 <= bins[2] <= 7


def test_uniform_5_covers_range():
    bins = split_uniform_5([-3.0, 0.0, 3.0])
    assert list(bins) == [1, 3, 5]


def test_round_7_is_not_uniform_7():
    # 2.6 rounds to 3 → class 6, but sits in the last uniform bin (7) or 6
    # depending on 6/7 edges. The two helpers must be allowed to disagree.
    y = np.array([2.6])
    if int(split_round_7(y)[0]) != int(split_uniform_7(y)[0]):
        return
    y = np.array([-0.4])
    assert int(split_round_7(y)[0]) != int(split_uniform_7(y)[0])


def test_regression_constant_pred_corr_is_zero():
    y = np.array([0.0, 1.0, 2.0])
    metrics = regression_metrics(y, np.ones_like(y))
    assert metrics["Corr"] == 0.0
    assert metrics["MAE"] == pytest.approx(2.0 / 3.0)


def test_format_metrics_row_has_six_numeric_cells():
    y = np.array([1.0, -1.0])
    row = format_metrics_row("demo", compute_affect_metrics(y, y))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[0] == "demo"
    assert len(cells) == 7
