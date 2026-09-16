"""Forward-shape tests for the fusion graphs (CPU, synthetic BERT widths)."""

from __future__ import annotations

import pytest
import torch

from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    GRUWithLinear,
    LateFusionTransformer,
    LowRankTensorFusion,
    LSTM,
    TensorFusion,
    TransformerSeq,
)
from synthetic import make_synthetic_batch


class TinyHParams:
    num_heads = 4
    layers = 1
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 16
    attn_mask = False
    output_dim = 1
    all_steps = False


@pytest.fixture(scope="module")
def batch():
    return make_synthetic_batch(batch_size=3, seq_len=8, embedding="bert", seed=0)


def _finite(tensor: torch.Tensor) -> None:
    assert torch.is_tensor(tensor)
    assert torch.isfinite(tensor).all()


def test_concat_early_shape(batch):
    fused = ConcatEarly()(batch.as_list())
    assert fused.shape == (3, 8, 35 + 74 + 768)
    _finite(fused)


def test_concat_late_shape(batch):
    encoded = [
        LSTM(35, 16, has_padding=False)(batch.visual),
        LSTM(74, 16, has_padding=False)(batch.audio),
        LSTM(768, 32, has_padding=False)(batch.text),
    ]
    fused = ConcatLate()(encoded)
    assert fused.shape == (3, 16 + 16 + 32)
    _finite(fused)


def test_low_rank_tensor_fusion_shape(batch):
    encoded = [
        GRUWithLinear(35, 16, 8, has_padding=True)([batch.visual, batch.lengths]),
        GRUWithLinear(74, 16, 8, has_padding=True)([batch.audio, batch.lengths]),
        GRUWithLinear(768, 32, 16, has_padding=True)([batch.text, batch.lengths]),
    ]
    fused = LowRankTensorFusion([8, 8, 16], 12, rank=3)(encoded)
    assert fused.shape == (3, 12)
    _finite(fused)


def test_tensor_fusion_shape(batch):
    encoded = [
        GRUWithLinear(35, 8, 4, has_padding=True)([batch.visual, batch.lengths]),
        GRUWithLinear(74, 8, 4, has_padding=True)([batch.audio, batch.lengths]),
        GRUWithLinear(768, 16, 4, has_padding=True)([batch.text, batch.lengths]),
    ]
    fused = TensorFusion()(encoded)
    # (4+1) * (4+1) * (4+1) = 125
    assert fused.shape == (3, 125)
    _finite(fused)


def test_transformer_early_shape(batch):
    fused = EarlyFusionTransformer(n_features=batch.total_feature_dim())(batch.as_list())
    assert fused.dim() == 2
    assert fused.size(0) == 3
    _finite(fused)


def test_transformer_late_shape(batch):
    encoded = [
        TransformerSeq(35, 16)(batch.visual),
        TransformerSeq(74, 16)(batch.audio),
        TransformerSeq(768, 32)(batch.text),
    ]
    assert encoded[0].shape == (3, 8, 16)
    in_dim = sum(t.size(-1) for t in encoded)
    fused = LateFusionTransformer(in_dim=in_dim, embed_dim=32)(encoded)
    assert fused.dim() == 2
    assert fused.size(0) == 3
    _finite(fused)


def test_gmtm_logit_shape(batch):
    model = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=TinyHParams)
    model.eval()
    with torch.no_grad():
        out = model(batch.as_list())
    assert out.shape == (3, 1)
    _finite(out)
