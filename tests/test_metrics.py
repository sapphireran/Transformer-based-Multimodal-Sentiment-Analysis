"""Pin the MOSI/MOSEI metric protocol used in the recorded CSVs."""

from __future__ import annotations

import numpy as np
import torch

from train_and_test import eval_affect, split_uniform_5, split_uniform_7


def test_acc7_edges_are_uniform_on_minus3_to_3():
    # digitize is 1-based; values exactly on an interior edge fall into the
    # *next* bin because right=False. The last edge (3.0) is clipped to 7.
    sample = np.array([-3.0, -3.0 + 6 / 7 - 1e-6, 0.0, 3.0])
    codes = split_uniform_7(sample)
    assert codes[0] == 1
    assert codes[1] == 1
    assert 1 <= codes[2] <= 7
    assert codes[3] == 7
    assert set(split_uniform_7(np.linspace(-3, 3, 70))).issubset(set(range(1, 8)))


def test_acc5_has_five_labels():
    codes = split_uniform_5(np.linspace(-3, 3, 50))
    assert set(codes).issubset(set(range(1, 6)))
    assert split_uniform_5(np.array([-3.0]))[0] == 1
    assert split_uniform_5(np.array([3.0]))[0] == 5


def test_zero_is_in_the_center_acc7_bin_not_its_own_class():
    """Uniform Acc-7 is not the same as rounding to {-3,...,3}."""
    zero = split_uniform_7(np.array([0.0]))[0]
    rounded_zero_would_be = 0
    assert zero != rounded_zero_would_be
    assert zero in (3, 4)  # the two bins that touch 0 under 6/7 width


def test_eval_affect_drops_exact_zeros_by_default():
    truth = torch.tensor([[-1.0], [0.0], [1.0]])
    pred = torch.tensor([[-0.2], [0.8], [0.4]])
    f1, acc = eval_affect(truth, pred, exclude_zero=True)
    # Only the first and last items remain: both polarity-correct → acc 1.
    assert acc == 1.0
    assert f1 == 1.0


def test_eval_affect_keeps_zeros_when_asked():
    truth = torch.tensor([[0.0], [1.0]])
    pred = torch.tensor([[-0.1], [0.4]])
    # y=0 is not > 0, so it is the negative class; pred=-0.1 is also negative.
    f1, acc = eval_affect(truth, pred, exclude_zero=False)
    assert acc == 1.0
    assert 0.0 <= f1 <= 1.0
