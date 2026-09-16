import numpy as np

from examples.eval_metrics import (
    eval_affect,
    evaluate,
    split_uniform_5,
    split_uniform_7,
)
from examples.metrics_walkthrough import GOLD, PRED


def test_uniform_7_edges():
    # -3 is the first edge → bin 1; 3 clips to bin 7.
    assert split_uniform_7([-3.0, 0.0, 3.0]).tolist() == [1, 4, 7]


def test_uniform_5_edges():
    assert split_uniform_5([-3.0, 0.0, 3.0]).tolist() == [1, 3, 5]


def test_eval_affect_drops_zero_and_checks_sign():
    f1, acc = eval_affect(GOLD, PRED, exclude_zero=True)
    # Four non-zero gold clips; one sign error (-0.2 vs +0.4).
    assert acc == 0.75
    assert 0.0 < f1 <= 1.0


def test_eval_affect_keeps_zero_when_asked():
    _, acc = eval_affect(GOLD, PRED, exclude_zero=False)
    # gold >0: F F F T T ; pred >0: F T T T T → 3/5
    assert acc == 0.6


def test_walkthrough_mae():
    metrics = evaluate(GOLD, PRED)
    assert abs(metrics["MAE"] - float(np.mean(np.abs(GOLD - PRED)))) < 1e-12
    assert metrics["Acc2"] == 0.75


def test_perfect_predictions_are_ones():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    metrics = evaluate(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["Acc2"] == 1.0
    assert metrics["Acc7_uniform"] == 1.0
    assert metrics["Acc5_uniform"] == 1.0
    assert abs(metrics["Corr"] - 1.0) < 1e-9
