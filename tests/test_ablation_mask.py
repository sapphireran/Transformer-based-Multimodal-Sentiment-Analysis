"""Zero-mask ablation keeps GMTM 3-wide and is a no-op when nothing is dropped."""

from __future__ import annotations

import torch

from ablation_zero_mask import COMBOS, TinyHParams, mask_streams
from common import dummy_modalities
from models import GatedMultiTransfomerModel


def test_mask_zeros_only_dropped_streams():
    visual, audio, text, _ = dummy_modalities(batch=2, seq_len=5, seed=9)
    masked = mask_streams([visual, audio, text], ["text"])
    assert torch.equal(masked[2], text)
    assert torch.count_nonzero(masked[0]) == 0
    assert torch.count_nonzero(masked[1]) == 0
    assert masked[0].shape == visual.shape
    assert masked[1].shape == audio.shape


def test_full_combo_is_identity_mask():
    streams = list(dummy_modalities(batch=2, seq_len=5, seed=10)[:3])
    assert all(torch.equal(a, b) for a, b in zip(mask_streams(streams, COMBOS[-1]), streams))


def test_same_weights_full_mask_matches_unmasked_forward():
    streams = list(dummy_modalities(batch=2, seq_len=6, seed=11)[:3])
    model = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=TinyHParams)
    model.eval()
    with torch.no_grad():
        a = model(streams)
        b = model(mask_streams(streams, ["text", "audio", "visual"]))
    assert torch.allclose(a, b)
    assert a.shape == (2, 1)
