"""CPU shape checks for GMTM and the fusion modules used in the sweeps."""

from __future__ import annotations

import torch

from common import BERT_DIM, dummy_modalities
from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    Identity,
    LateFusionTransformer,
    LowRankTensorFusion,
    MLP,
    TensorFusion,
)
from train_and_test import MultiFramework


class TinyGmtmHParams:
    num_heads = 4
    layers = 1
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 16
    attn_mask = True
    output_dim = 1
    all_steps = False


def test_gmtm_forward_shape():
    visual, audio, text, _ = dummy_modalities(batch=3, seq_len=8, text_dim=BERT_DIM, seed=3)
    model = GatedMultiTransfomerModel(3, [35, 74, BERT_DIM], hyp_params=TinyGmtmHParams)
    model.eval()
    with torch.no_grad():
        out = model([visual, audio, text])
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_concat_and_tensor_fusion_shapes():
    batch, time = 2, 6
    visual = torch.randn(batch, time, 35)
    audio = torch.randn(batch, time, 74)
    text = torch.randn(batch, time, BERT_DIM)
    early = ConcatEarly()([visual, audio, text])
    assert early.shape == (batch, time, 35 + 74 + BERT_DIM)

    late = ConcatLate()([visual, audio, text])
    assert late.shape == (batch, time * (35 + 74 + BERT_DIM))

    tfn = TensorFusion()([torch.randn(batch, 19), torch.randn(batch, 39), torch.randn(batch, 159)])
    assert tfn.shape == (batch, 20 * 40 * 160)

    lmf = LowRankTensorFusion([32, 64, 256], output_dim=256, rank=8)
    lmf.eval()
    with torch.no_grad():
        out = lmf([torch.randn(batch, 32), torch.randn(batch, 64), torch.randn(batch, 256)])
    assert out.shape == (batch, 256)


def test_transformer_fusion_wrappers_return_clip_vectors():
    batch, time = 2, 5
    streams = [
        torch.randn(batch, time, 35),
        torch.randn(batch, time, 74),
        torch.randn(batch, time, BERT_DIM),
    ]
    early = EarlyFusionTransformer(n_features=35 + 74 + BERT_DIM)
    late = LateFusionTransformer(in_dim=64 + 128 + 1024)
    late_in = [torch.randn(batch, time, 64), torch.randn(batch, time, 128), torch.randn(batch, time, 1024)]
    early.eval()
    late.eval()
    with torch.no_grad():
        e = early(streams)
        l = late(late_in)
    assert e.shape == (batch, early.embed_dim)
    assert l.shape == (batch, late.embed_dim)


def test_multiframework_gmtm_matches_bare_fusion():
    streams = list(dummy_modalities(batch=2, seq_len=6, seed=4)[:3])
    fusion = GatedMultiTransfomerModel(3, [35, 74, BERT_DIM], hyp_params=TinyGmtmHParams)
    wrapped = MultiFramework([Identity(), Identity(), Identity()], fusion, Identity())
    fusion.eval()
    wrapped.eval()
    # Copy weights so both graphs are identical.
    wrapped.fuse.load_state_dict(fusion.state_dict())
    with torch.no_grad():
        a = fusion(streams)
        b = wrapped(streams)
    assert torch.allclose(a, b)
    assert b.shape == (2, 1)


def test_late_concat_mlp_head_shape():
    visual, audio, text, _ = dummy_modalities(batch=2, seq_len=6, seed=5)
    pooled = [visual.mean(1), audio.mean(1), text.mean(1)]
    fused = ConcatLate()(pooled)
    head = MLP(fused.size(1), 32, 1)
    out = head(fused)
    assert out.shape == (2, 1)
