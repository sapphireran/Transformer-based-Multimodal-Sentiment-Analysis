"""Synthetic batch shapes must match the real MOSI/MOSEI loader contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.synthetic import (
    AUDIO_DIM,
    BERT_SPEC,
    BERT_TEXT_DIM,
    GLOVE_SPEC,
    VISION_DIM,
    ablate_modalities,
    iter_ablation_sets,
    make_batch,
)


def test_bert_batch_shapes():
    batch = make_batch(batch_size=5, spec=BERT_SPEC, seed=0)
    assert batch.shapes() == {
        "vision": (5, 50, VISION_DIM),
        "audio": (5, 50, AUDIO_DIM),
        "text": (5, 50, BERT_TEXT_DIM),
        "labels": (5, 1),
    }
    assert batch.labels.min() >= -3.0
    assert batch.labels.max() <= 3.0


def test_glove_text_width():
    batch = make_batch(batch_size=2, spec=GLOVE_SPEC, seed=1)
    assert batch.text.shape[-1] == 300
    assert batch.spec.total_early == 35 + 74 + 300


def test_seed_is_deterministic():
    a = make_batch(4, BERT_SPEC, seed=11)
    b = make_batch(4, BERT_SPEC, seed=11)
    np.testing.assert_array_equal(a.text, b.text)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_ablate_text_only_zeros_others():
    src = make_batch(3, BERT_SPEC, seed=2)
    out = ablate_modalities(src, ["text"])
    assert np.allclose(out.text, src.text)
    assert np.allclose(out.audio, 0)
    assert np.allclose(out.vision, 0)
    np.testing.assert_array_equal(out.labels, src.labels)


def test_ablate_rejects_unknown_name():
    src = make_batch(1, BERT_SPEC, seed=0)
    with pytest.raises(ValueError, match="unknown"):
        ablate_modalities(src, ["face"])


def test_seven_ablation_sets():
    sets = list(iter_ablation_sets())
    assert len(sets) == 7
    assert sets[0] == ["text"]
    assert sets[-1] == ["text", "audio", "visual"]
