import numpy as np
import pytest

from examples.metrics_lib import (
    eval_affect,
    evaluate_affect_batch,
    split_uniform_5,
    split_uniform_7,
    uniform_edges,
)


def test_uniform_edges_span_minus_three_to_three():
    e7 = uniform_edges(7)
    e5 = uniform_edges(5)
    assert e7[0] == pytest.approx(-3.0)
    assert e7[-1] == pytest.approx(3.0)
    assert len(e7) == 8
    assert e5[0] == pytest.approx(-3.0)
    assert e5[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(e7), e7[1] - e7[0])


def test_split_uniform_7_known_points():
    # digitize is left-closed; 0.0 lands in the center bin (4).
    # ±0.4 both sit inside the center bin (edges at ±6/7), so use ±1.5.
    values = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    bins = split_uniform_7(values)
    assert bins[0] == 1
    assert bins[-1] == 7
    assert bins[2] == 4
    assert bins[1] < bins[3]


def test_split_uniform_5_monotonic():
    xs = np.linspace(-3, 3, 11)
    bins = split_uniform_5(xs)
    assert bins.min() == 1
    assert bins.max() == 5
    assert np.all(np.diff(bins) >= 0)


def test_eval_affect_drops_true_zeros():
    y = np.array([-1.0, 0.0, 0.0, 2.0])
    yhat = np.array([-0.5, 4.0, -4.0, 1.0])
    f1, acc = eval_affect(y, yhat, exclude_zero=True)
    # only the first and last items remain; both signs match
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_evaluate_perfect_and_flipped():
    y = np.array([-2.0, -0.5, 0.5, 2.0])
    perfect = evaluate_affect_batch(y, y)
    assert perfect["MAE"] == pytest.approx(0.0)
    assert perfect["Acc7_uniform"] == pytest.approx(1.0)
    assert perfect["Corr"] == pytest.approx(1.0)
    flipped = evaluate_affect_batch(y, -y)
    assert flipped["Corr"] == pytest.approx(-1.0)
    assert flipped["Acc2"] == pytest.approx(0.0)
