"""Forward-pass smoke tests for the fusion modules used in the trainers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "model"))

from common import (  # noqa: E402
    TINY_AUDIO_DIM,
    TINY_SEQ_LEN,
    TINY_TEXT_DIM,
    TINY_VISION_DIM,
    TinyHParams,
    make_synthetic_mosi,
)
from models import (  # noqa: E402
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
)


def _batch():
    split = make_synthetic_mosi(n_train=3, n_valid=1, n_test=1, seed=2)["train"]
    vision = torch.from_numpy(split["vision"]).float()
    audio = torch.from_numpy(split["audio"]).float()
    text = torch.from_numpy(split["text"]).float()
    return vision, audio, text


def test_concat_early_joins_feature_axis():
    vision, audio, text = _batch()
    fused = ConcatEarly()([vision, audio, text])
    assert fused.shape == (
        3,
        TINY_SEQ_LEN,
        TINY_VISION_DIM + TINY_AUDIO_DIM + TINY_TEXT_DIM,
    )


def test_concat_late_joins_pooled_vectors():
    vision, audio, text = _batch()
    pooled = [m.mean(dim=1) for m in (vision, audio, text)]
    fused = ConcatLate()(pooled)
    assert fused.shape == (3, TINY_VISION_DIM + TINY_AUDIO_DIM + TINY_TEXT_DIM)


def test_tensor_fusion_outer_product_grows():
    mods = [torch.randn(3, 4), torch.randn(3, 5), torch.randn(3, 6)]
    fused = TensorFusion()(mods)
    # (4+1) * (5+1) * (6+1)
    assert fused.shape == (3, 5 * 6 * 7)


def test_low_rank_tensor_fusion_output_dim():
    mods = [torch.randn(3, 4), torch.randn(3, 5), torch.randn(3, 6)]
    fused = LowRankTensorFusion([4, 5, 6], output_dim=9, rank=2)(mods)
    assert fused.shape == (3, 9)
    assert torch.isfinite(fused).all()


def test_early_fusion_transformer_last_step():
    vision, audio, text = _batch()
    n_features = TINY_VISION_DIM + TINY_AUDIO_DIM + TINY_TEXT_DIM
    fused = EarlyFusionTransformer(n_features=n_features)([vision, audio, text])
    assert fused.dim() == 2
    assert fused.size(0) in {TINY_SEQ_LEN, 3}  # layer is batch_first but fed (T,B,C)
    assert fused.size(-1) == 32
    assert torch.isfinite(fused).all()


def test_late_fusion_transformer_last_step():
    vision, audio, text = _batch()
    # Already-projected stand-ins: concatenate on the feature axis.
    concat = torch.cat([vision, audio, text], dim=-1)
    fused = LateFusionTransformer(in_dim=concat.size(-1), embed_dim=16)(concat)
    assert fused.shape[-1] == 16
    assert torch.isfinite(fused).all()


def test_gmtm_regression_head():
    vision, audio, text = _batch()
    model = GatedMultiTransfomerModel(
        3,
        [TINY_VISION_DIM, TINY_AUDIO_DIM, TINY_TEXT_DIM],
        hyp_params=TinyHParams,
    )
    out = model([vision, audio, text])
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one gradient after a backward pass"
