"""Packed vs max-pad collate helpers."""

from __future__ import annotations

import torch

from msa_lab.collate import collate_max_pad, collate_variable_length, summarize_collate
from msa_lab.synthetic import make_sentiment_batch


def test_packed_respects_lengths():
    batch = make_sentiment_batch(
        batch_size=4, seq_len=12, preset="toy", seed=8, variable_lengths=True, min_length=4
    )
    packed = collate_variable_length(batch)
    assert packed["style"] == "process_1_packed"
    assert packed["vision"].shape[0] == 4
    assert packed["vision"].shape[1] == int(batch.lengths.max())
    assert torch.equal(packed["lengths"], batch.lengths)


def test_max_pad_is_dense():
    batch = make_sentiment_batch(
        batch_size=3, seq_len=9, preset="toy", seed=8, variable_lengths=True, min_length=3
    )
    padded = collate_max_pad(batch, max_pad_num=15)
    assert padded["vision"].shape == (3, 15, 8)
    assert padded["audio"].shape == (3, 15, 12)
    assert padded["text"].shape == (3, 15, 16)
    assert "lengths" not in padded


def test_summarize_collate_rows():
    batch = make_sentiment_batch(
        batch_size=2, seq_len=6, preset="toy", seed=1, variable_lengths=True
    )
    rows = summarize_collate(collate_variable_length(batch), collate_max_pad(batch))
    names = [row[0] for row in rows]
    assert names == ["vision", "audio", "text", "lengths"]
