"""Unit tests for the shared MOSI / MOSEI scoring helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from metrics import (  # noqa: E402
    as_numpy,
    bin_intervals,
    compute_sentiment_metrics,
    eval_affect,
    format_metrics,
    split_uniform,
    split_uniform_5,
    split_uniform_7,
    uniform_bin_edges,
)


class TestAsNumpy:
    def test_flattens_nested_lists(self):
        assert as_numpy([[1.0], [2.0], [3.0]]).tolist() == [1.0, 2.0, 3.0]

    def test_preserves_signed_zeros_as_zero(self):
        values = as_numpy(np.array([0.0, -0.0, 1.0]))
        assert values[0] == 0.0
        assert values[1] == 0.0


class TestUniformBins:
    def test_seven_bin_edges_cover_closed_interval(self):
        # Each bin is 6/7 ≈ 0.857 wide. Values sit at the center of bins 1..7.
        centers = np.array([-3 + (i + 0.5) * (6 / 7) for i in range(7)])
        assert split_uniform_7(centers).tolist() == [1, 2, 3, 4, 5, 6, 7]

    def test_five_bin_edges_cover_closed_interval(self):
        centers = np.array([-3 + (i + 0.5) * (6 / 5) for i in range(5)])
        assert split_uniform_5(centers).tolist() == [1, 2, 3, 4, 5]

    def test_clip_extremes(self):
        assert split_uniform_7([-10.0, 10.0]).tolist() == [1, 7]
        assert split_uniform_5([-10.0, 10.0]).tolist() == [1, 5]

    def test_rejects_tiny_bin_count(self):
        with pytest.raises(ValueError):
            split_uniform([0.0], n_bins=1)

    def test_uniform_bin_edges_length(self):
        edges = uniform_bin_edges(7)
        assert len(edges) == 8
        assert edges[0] == pytest.approx(-3.0)
        assert edges[-1] == pytest.approx(3.0)

    def test_bin_intervals_are_contiguous(self):
        intervals = bin_intervals(5)
        assert [idx for idx, _left, _right in intervals] == [1, 2, 3, 4, 5]
        for i in range(len(intervals) - 1):
            assert intervals[i][2] == pytest.approx(intervals[i + 1][1])


class TestEvalAffect:
    def test_perfect_sign_prediction(self):
        truth = np.array([-2.0, -1.0, 1.0, 2.0])
        pred = np.array([-0.1, -0.2, 0.3, 0.4])
        f1, acc = eval_affect(truth, pred, exclude_zero=True)
        assert acc == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_excludes_neutral_labels(self):
        truth = np.array([-1.0, 0.0, 1.0])
        pred = np.array([-0.5, 3.0, 0.5])
        f1, acc = eval_affect(truth, pred, exclude_zero=True)
        assert acc == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_keeps_neutral_when_requested(self):
        truth = np.array([0.0, 1.0])
        pred = np.array([0.2, 0.2])
        # 0 is not > 0, so the first sample is a false positive.
        _f1, acc = eval_affect(truth, pred, exclude_zero=False)
        assert acc == pytest.approx(0.5)


class TestComputeSentimentMetrics:
    def test_perfect_regression(self):
        values = np.array([-2.5, -1.0, 0.5, 2.0])
        scores = compute_sentiment_metrics(values, values)
        assert scores["MAE"] == pytest.approx(0.0)
        assert scores["MSE"] == pytest.approx(0.0)
        assert scores["Corr"] == pytest.approx(1.0)
        assert scores["Acc7_uniform"] == pytest.approx(1.0)
        assert scores["Acc5_uniform"] == pytest.approx(1.0)
        assert scores["Acc2"] == pytest.approx(1.0)
        assert scores["F1"] == pytest.approx(1.0)

    def test_constant_prediction_has_zero_corr(self):
        truth = np.array([-2.0, -1.0, 1.0, 2.0])
        pred = np.array([0.1, 0.1, 0.1, 0.1])
        scores = compute_sentiment_metrics(truth, pred)
        assert scores["Corr"] == pytest.approx(0.0)
        assert scores["MAE"] == pytest.approx(np.mean(np.abs(truth - pred)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_sentiment_metrics([1.0, 2.0], [1.0])

    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            compute_sentiment_metrics([], [])

    def test_format_metrics_includes_names(self):
        text = format_metrics({"MAE": 0.5, "Acc2": 0.8})
        assert "MAE" in text
        assert "0.5000" in text


class TestLegacyParity:
    """The extracted helpers must match the original digitize + clip recipe."""

    def test_matches_original_seven_bin_formula(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(-3.5, 3.5, size=64)
        step = 6.0 / 7.0
        edges = [-3.0 + i * step for i in range(8)]
        expected = np.clip(np.digitize(data, edges, right=False), 1, 7)
        assert np.array_equal(split_uniform_7(data), expected)

    def test_matches_original_five_bin_formula(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(-3.5, 3.5, size=64)
        step = 6.0 / 5.0
        edges = [-3.0 + i * step for i in range(6)]
        expected = np.clip(np.digitize(data, edges, right=False), 1, 5)
        assert np.array_equal(split_uniform_5(data), expected)

    def test_corr_is_finite_for_noisy_linear_target(self):
        rng = np.random.default_rng(2)
        truth = rng.uniform(-3, 3, size=128)
        pred = truth + rng.normal(0, 0.1, size=128)
        scores = compute_sentiment_metrics(truth, pred)
        assert math.isfinite(scores["Corr"])
        assert scores["Corr"] > 0.9
        assert scores["MAE"] < 0.2
