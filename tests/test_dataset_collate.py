"""Affectdataset + collate functions on a tiny in-memory split."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import AUDIO_DIM, BERT_DIM, VISUAL_DIM
from data.get_dataloader import Affectdataset, _process_1, _process_2


def _split(n=6, seq_len=10):
    rng = np.random.default_rng(11)
    text = rng.normal(size=(n, seq_len, BERT_DIM)).astype(np.float32)
    text[:, 0, 0] = 1.0
    return {
        "vision": rng.normal(size=(n, seq_len, VISUAL_DIM)).astype(np.float32),
        "audio": rng.normal(size=(n, seq_len, AUDIO_DIM)).astype(np.float32),
        "text": text,
        "labels": rng.uniform(-2, 2, size=(n, 1, 1)).astype(np.float32),
    }


def test_packed_collate_shapes():
    ds = Affectdataset(_split(), flatten_time_series=False, aligned=True, max_pad=False)
    loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=_process_1)
    streams, lengths, indices, labels = next(iter(loader))
    assert len(streams) == 3
    assert streams[0].shape[0] == 3
    assert streams[0].shape[-1] == VISUAL_DIM
    assert streams[1].shape[-1] == AUDIO_DIM
    assert streams[2].shape[-1] == BERT_DIM
    assert len(lengths) == 3
    assert indices.shape == (3, 1)
    assert labels.shape == (3, 1)


def test_padded_collate_is_rectangular():
    ds = Affectdataset(
        _split(), flatten_time_series=False, aligned=True, max_pad=True, max_pad_num=7
    )
    loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=_process_2)
    visual, audio, text, labels = next(iter(loader))
    assert visual.shape == (3, 7, VISUAL_DIM)
    assert audio.shape == (3, 7, AUDIO_DIM)
    assert text.shape == (3, 7, BERT_DIM)
    assert labels.shape == (3, 1)


def test_minus_inf_audio_is_zeroed():
    split = _split(n=1, seq_len=5)
    split["audio"][0, 1, 2] = -np.inf
    ds = Affectdataset(split, flatten_time_series=False, aligned=True, max_pad=True, max_pad_num=5)
    audio = ds[0][1]
    assert torch.isfinite(audio).all()
    assert audio[1, 2].item() == 0.0


def test_aligned_slice_skips_leading_zero_text():
    split = _split(n=1, seq_len=8)
    split["text"][0, :4, :] = 0.0
    split["text"][0, 4, 0] = 1.0
    ds = Affectdataset(split, flatten_time_series=False, aligned=True, max_pad=False)
    vision, audio, text, _index, _label = ds[0]
    assert text.shape[0] == 4
    assert vision.shape[0] == 4
    assert audio.shape[0] == 4
