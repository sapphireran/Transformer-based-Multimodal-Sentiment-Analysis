"""Tests for the synthetic MOSI/MOSEI-shaped batches."""

from __future__ import annotations

import pytest
import torch

from synthetic import (
    AUDIO_DIM,
    BERT_TEXT_DIM,
    GLOVE_TEXT_DIM,
    VISUAL_DIM,
    make_synthetic_batch,
    make_synthetic_split,
    modality_dims,
    text_dim_for,
    zero_out_modalities,
)


def test_text_dim_and_modality_dims():
    assert text_dim_for("bert") == BERT_TEXT_DIM
    assert text_dim_for("glove") == GLOVE_TEXT_DIM
    assert modality_dims("bert") == (VISUAL_DIM, AUDIO_DIM, BERT_TEXT_DIM)
    assert modality_dims("glove") == (VISUAL_DIM, AUDIO_DIM, GLOVE_TEXT_DIM)
    with pytest.raises(ValueError):
        text_dim_for("word2vec")  # type: ignore[arg-type]


@pytest.mark.parametrize("embedding, t_dim", [("bert", BERT_TEXT_DIM), ("glove", GLOVE_TEXT_DIM)])
def test_batch_shapes_and_label_range(embedding, t_dim):
    batch = make_synthetic_batch(batch_size=5, seq_len=9, embedding=embedding, seed=4)
    assert batch.visual.shape == (5, 9, VISUAL_DIM)
    assert batch.audio.shape == (5, 9, AUDIO_DIM)
    assert batch.text.shape == (5, 9, t_dim)
    assert batch.labels.shape == (5, 1)
    assert batch.lengths.tolist() == [9] * 5
    assert float(batch.labels.min()) >= -3.0
    assert float(batch.labels.max()) <= 3.0
    assert batch.total_feature_dim() == VISUAL_DIM + AUDIO_DIM + t_dim
    assert len(batch.as_list()) == 3


def test_reproducible_seed():
    a = make_synthetic_batch(batch_size=3, seq_len=4, seed=99)
    b = make_synthetic_batch(batch_size=3, seq_len=4, seed=99)
    assert torch.equal(a.text, b.text)
    assert torch.equal(a.labels, b.labels)


def test_zero_out_modalities_keeps_labels():
    batch = make_synthetic_batch(batch_size=2, seq_len=4, seed=1)
    dropped = zero_out_modalities(batch, keep=("text",))
    assert torch.equal(dropped.labels, batch.labels)
    assert torch.equal(dropped.text, batch.text)
    assert torch.count_nonzero(dropped.audio) == 0
    assert torch.count_nonzero(dropped.visual) == 0
    with pytest.raises(ValueError):
        zero_out_modalities(batch, keep=("smell",))


def test_make_synthetic_split_loaders():
    loaders = make_synthetic_split(
        n_train=10, n_valid=6, n_test=6, batch_size=4, seq_len=5, seed=2
    )
    assert set(loaders) == {"train", "valid", "test"}
    visual, audio, text, labels = next(iter(loaders["train"]))
    assert visual.shape[-1] == VISUAL_DIM
    assert audio.shape[-1] == AUDIO_DIM
    assert text.shape[-1] == BERT_TEXT_DIM
    assert labels.shape[-1] == 1
