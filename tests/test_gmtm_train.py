"""GMTM can overfit one synthetic batch on CPU."""

from __future__ import annotations

from msa_lab.toy_train import train_gated_transformer


def test_toy_mae_drops():
    result = train_gated_transformer(steps=12, batch_size=16, seq_len=10, seed=21)
    assert result.steps == 12
    assert result.final_mae < result.initial_mae
    assert result.history[-1] <= result.history[0] + 1e-6
    assert result.final_mae < 1.5
