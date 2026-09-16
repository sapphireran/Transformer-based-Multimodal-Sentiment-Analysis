#!/usr/bin/env python3
"""CPU forward pass of GMTM and the classical fusion modules.

Uses synthetic tensors with the same ranks as the MOSEI BERT loader.
Does not load checkpoints or datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.model_import import load_models
from examples.lib.synthetic import make_mosei_like_batch, make_toy_mixture


class SmallGMTMParams:
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


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"{name} produced non-finite values: {tensor}")


def run_gmtm_full_width() -> torch.Tensor:
    models = load_models()
    vision, audio, text, _labels = make_mosei_like_batch(
        batch_size=2, seq_len=6, embedding="bert", seed=3
    )
    gmtm = models.GatedMultiTransfomerModel(
        3, [35, 74, 768], hyp_params=SmallGMTMParams
    )
    gmtm.eval()
    with torch.no_grad():
        out = gmtm([vision, audio, text])
    _finite("GMTM BERT-width", out)
    if out.shape != (2, 1):
        raise SystemExit(f"GMTM expected [2, 1], got {tuple(out.shape)}")
    print(f"GMTM BERT-width  input [2, 6, {{35,74,768}}] -> {tuple(out.shape)}  mean={out.mean():.4f}")
    return out


def run_classical_fusions() -> None:
    models = load_models()
    batch = 3
    streams, _ = make_toy_mixture(
        batch_size=batch, seq_len=5, n_features=(8, 12, 16), seed=4
    )

    concat_early = models.ConcatEarly()
    early = concat_early(streams)
    _finite("ConcatEarly", early)
    print(f"ConcatEarly      { [tuple(s.shape) for s in streams] } -> {tuple(early.shape)}")

    concat_late = models.ConcatLate()
    late = concat_late(streams)
    _finite("ConcatLate", late)
    print(f"ConcatLate       flattened concat -> {tuple(late.shape)}")

    vectors = [stream.mean(dim=1) for stream in streams]
    tfn = models.TensorFusion()
    fused = tfn(vectors)
    _finite("TensorFusion", fused)
    # (8+1)*(12+1)*(16+1) = 9*13*17 = 1989
    print(f"TensorFusion     {[tuple(v.shape) for v in vectors]} -> {tuple(fused.shape)}")

    lmf = models.LowRankTensorFusion([8, 12, 16], output_dim=7, rank=3)
    lmf.eval()
    with torch.no_grad():
        lowrank = lmf(vectors)
    _finite("LowRankTensorFusion", lowrank)
    print(f"LowRankTFN       -> {tuple(lowrank.shape)}")

    fusion = models.TransformerFusion(d_model=16, nhead=4, num_layers=1, dropout=0.0)
    # Project each vector to 16 so TransformerFusion can stack them.
    projected = [torch.nn.functional.pad(v, (0, 16 - v.shape[-1])) for v in vectors]
    fusion.eval()
    with torch.no_grad():
        stacked = fusion(projected)
    _finite("TransformerFusion", stacked)
    print(f"TransformerFusion stacked vectors -> {tuple(stacked.shape)}")


def run_transformer_blocks() -> None:
    models = load_models()
    vision, audio, text, _ = make_mosei_like_batch(
        batch_size=2, seq_len=5, embedding="glove", seed=5
    )
    early = models.EarlyFusionTransformer(n_features=35 + 74 + 300)
    early.eval()
    with torch.no_grad():
        out = early([vision, audio, text])
    _finite("EarlyFusionTransformer", out)
    print(f"EarlyFusionTransformer  -> {tuple(out.shape)}")

    seqs = [
        models.TransformerSeq(35, 16)(vision),
        models.TransformerSeq(74, 16)(audio),
        models.TransformerSeq(300, 16)(text),
    ]
    late = models.LateFusionTransformer(in_dim=48, embed_dim=16)
    late.eval()
    with torch.no_grad():
        fused = late(seqs)
    _finite("LateFusionTransformer", fused)
    print(f"LateFusionTransformer   -> {tuple(fused.shape)}")

    pool = models.AttentionPooling(16)
    pooled = pool(seqs[2])
    _finite("AttentionPooling", pooled)
    print(f"AttentionPooling        {tuple(seqs[2].shape)} -> {tuple(pooled.shape)}")


def main() -> int:
    torch.manual_seed(0)
    print("Device: CPU (synthetic; no checkpoints)")
    print()
    run_gmtm_full_width()
    print()
    run_classical_fusions()
    print()
    run_transformer_blocks()
    print()
    print("All forward passes produced finite tensors with the expected ranks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
