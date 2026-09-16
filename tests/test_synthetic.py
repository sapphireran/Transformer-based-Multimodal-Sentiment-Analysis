"""Synthetic batch shapes and ablation zeroing."""

from __future__ import annotations

import pytest
import torch

from msa_lab.synthetic import FEATURE_DIMS, make_sentiment_batch, zero_modalities


def test_toy_shapes():
    batch = make_sentiment_batch(batch_size=5, seq_len=9, preset="toy", seed=0)
    assert batch.visual.shape == (5, 9, 8)
    assert batch.audio.shape == (5, 9, 12)
    assert batch.text.shape == (5, 9, 16)
    assert batch.labels.shape == (5, 1)
    assert torch.all(batch.labels >= -3) and torch.all(batch.labels <= 3)


def test_named_packs_match_feature_table():
    for preset, dims in FEATURE_DIMS.items():
        batch = make_sentiment_batch(batch_size=2, seq_len=4, preset=preset, seed=1)
        assert batch.dims() == dims


def test_forced_label_anchors():
    batch = make_sentiment_batch(batch_size=4, seq_len=4, preset="toy", seed=99)
    assert float(batch.labels[0]) == pytest.approx(2.4)
    assert float(batch.labels[1]) == pytest.approx(-1.8)
    assert float(batch.labels[2]) == pytest.approx(0.0)
    assert float(batch.labels[3]) == pytest.approx(0.7)


def test_variable_lengths_zero_tail():
    batch = make_sentiment_batch(
        batch_size=3, seq_len=10, preset="toy", seed=4, variable_lengths=True, min_length=3
    )
    assert int(batch.lengths.min()) >= 3
    assert int(batch.lengths.max()) <= 10
    for index, length in enumerate(batch.lengths.tolist()):
        if length < 10:
            assert torch.all(batch.visual[index, length:] == 0)


def test_zero_modalities_drops_energy():
    full = make_sentiment_batch(batch_size=4, seq_len=6, preset="toy", seed=5)
    dropped = zero_modalities(full, ["audio", "visual"])
    assert torch.count_nonzero(dropped.audio) == 0
    assert torch.count_nonzero(dropped.visual) == 0
    assert torch.equal(dropped.text, full.text)
    assert torch.equal(dropped.labels, full.labels)


def test_unknown_preset_raises():
    try:
        make_sentiment_batch(preset="not-a-pack")
    except KeyError as error:
        assert "not-a-pack" in str(error)
    else:
        raise AssertionError("expected KeyError")
