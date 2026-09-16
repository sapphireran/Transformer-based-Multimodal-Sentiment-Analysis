"""Synthetic MOSI pickle matches the schema the real loaders expect."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "model"))

from common import (  # noqa: E402
    MOSI_AUDIO_DIM,
    MOSI_BERT_DIM,
    MOSI_SEQ_LEN,
    MOSI_VISION_DIM,
    TINY_AUDIO_DIM,
    TINY_SEQ_LEN,
    TINY_TEXT_DIM,
    TINY_VISION_DIM,
    make_synthetic_mosi,
    write_synthetic_pickle,
)


def test_tiny_bundle_shapes():
    bundle = make_synthetic_mosi(n_train=6, n_valid=2, n_test=2, seed=3)
    assert set(bundle) == {"train", "valid", "test"}
    train = bundle["train"]
    assert train["vision"].shape == (6, TINY_SEQ_LEN, TINY_VISION_DIM)
    assert train["audio"].shape == (6, TINY_SEQ_LEN, TINY_AUDIO_DIM)
    assert train["text"].shape == (6, TINY_SEQ_LEN, TINY_TEXT_DIM)
    assert train["labels"].shape == (6, 1, 1)
    assert len(train["id"]) == 6
    assert np.isfinite(train["labels"]).all()
    assert train["labels"].min() >= -3.0
    assert train["labels"].max() <= 3.0
    # Affectdataset alignment looks for a non-zero text frame.
    assert not np.allclose(train["text"][:, 0, :], 0.0)


def test_full_width_matches_mosi_bert():
    bundle = make_synthetic_mosi(
        n_train=2,
        n_valid=1,
        n_test=1,
        seq_len=MOSI_SEQ_LEN,
        vision_dim=MOSI_VISION_DIM,
        audio_dim=MOSI_AUDIO_DIM,
        text_dim=MOSI_BERT_DIM,
        seed=1,
    )
    vision = bundle["test"]["vision"]
    assert vision.shape == (1, MOSI_SEQ_LEN, MOSI_VISION_DIM)
    assert bundle["test"]["audio"].shape[-1] == MOSI_AUDIO_DIM
    assert bundle["test"]["text"].shape[-1] == MOSI_BERT_DIM


def test_write_and_reload(tmp_path: Path):
    path = write_synthetic_pickle(tmp_path / "toy.pkl", n_train=4, n_valid=2, n_test=2)
    pytest.importorskip("torch")
    from data.get_dataloader import get_dataloader

    train, valid, test = get_dataloader(
        str(path),
        batch_size=2,
        max_seq_len=TINY_SEQ_LEN,
        max_pad=True,
        num_workers=0,
        data_type="mosi",
        train_shuffle=False,
    )
    vision, audio, text, labels = next(iter(train))
    assert vision.shape[0] == 2
    assert vision.shape[-1] == TINY_VISION_DIM
    assert audio.shape[-1] == TINY_AUDIO_DIM
    assert text.shape[-1] == TINY_TEXT_DIM
    assert labels.shape[-1] == 1
    assert len(train.dataset) == 4
    assert len(valid.dataset) == 2
    assert len(test.dataset) == 2
