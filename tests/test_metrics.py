import numpy as np
from sklearn.metrics import accuracy_score

from train_and_test import eval_affect, split_uniform_5, split_uniform_7


def test_uniform_7_edges_cover_closed_interval():
    # Left edge of the first bin and right edge of the last bin.
    cats = split_uniform_7(np.array([-3.0, 3.0]))
    assert cats[0] == 1
    assert cats[1] == 7


def test_uniform_5_is_monotonic():
    values = np.linspace(-3.0, 3.0, 11)
    cats = split_uniform_5(values)
    assert cats.min() == 1
    assert cats.max() == 5
    assert np.all(np.diff(cats) >= 0)


def test_perfect_predictions_are_one():
    y = np.array([-2.0, -0.5, 0.4, 1.2, 2.5])
    acc7 = accuracy_score(split_uniform_7(y), split_uniform_7(y.copy()))
    acc5 = accuracy_score(split_uniform_5(y), split_uniform_5(y.copy()))
    f1, acc2 = eval_affect(y, y.copy(), exclude_zero=True)
    assert acc7 == 1.0
    assert acc5 == 1.0
    assert acc2 == 1.0
    assert f1 == 1.0


def test_eval_affect_drops_zero_labels():
    y = np.array([0.0, 0.0, 1.0, -1.0])
    yhat = np.array([2.0, -2.0, 1.5, -0.2])
    f1, acc2 = eval_affect(y, yhat, exclude_zero=True)
    # Only the last two gold scores count; both signs match.
    assert acc2 == 1.0
    assert f1 == 1.0


def test_eval_affect_counts_zeros_when_asked():
    y = np.array([0.0, 1.0])
    yhat = np.array([0.2, 1.0])
    # 0.2 > 0 so the zero gold is treated as a false positive when kept.
    _f1, acc2 = eval_affect(y, yhat, exclude_zero=False)
    assert acc2 == 0.5


def test_evaluate_metrics_example_is_stable():
    from evaluate_metrics import run

    result = run()
    assert 0.0 <= result["acc7"] <= 1.0
    assert 0.0 <= result["acc2"] <= 1.0
    assert result["notes"]["n_binary"] == 6
    # The third gold is negative, the third pred is positive.
    assert result["notes"]["sign_flip_at_index"] == 2
    assert result["mae"] > 0.0
