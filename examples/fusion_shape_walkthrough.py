#!/usr/bin/env python3
"""Print the rank contract of every fusion block used in the personal experiments.

This is the architecture.md shape table in executable form. Widths are the
MOSEI BERT defaults unless a module needs a smaller toy width to stay
readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.model_import import load_models


def _line(name: str, before: str, after: torch.Tensor) -> None:
    print(f"{name:<28} {before:<42} -> {tuple(after.shape)}")


def main() -> int:
    models = load_models()
    batch, time = 4, 50
    vision = torch.randn(batch, time, 35)
    audio = torch.randn(batch, time, 74)
    text = torch.randn(batch, time, 768)
    trio = [vision, audio, text]

    print("Batch convention: vision / audio / text = [B, T, 35/74/768]")
    print()

    _line("ConcatEarly", "[B,T,35]+[B,T,74]+[B,T,768]", models.ConcatEarly()(trio))
    _line("ConcatLate", "flatten then cat on dim=1", models.ConcatLate()(trio))

    vectors = [vision.mean(1), audio.mean(1), text.mean(1)]
    _line("TensorFusion (pooled)", "[B,35]x[B,74]x[B,768] + ones", models.TensorFusion()(vectors))

    lmf = models.LowRankTensorFusion([35, 74, 32], output_dim=16, rank=4)
    tiny_text = torch.randn(batch, 32)
    with torch.no_grad():
        _line(
            "LowRankTensorFusion",
            "[B,35],[B,74],[B,32] rank=4",
            lmf([vectors[0], vectors[1], tiny_text]),
        )

    early = models.EarlyFusionTransformer(n_features=877)
    with torch.no_grad():
        _line("EarlyFusionTransformer", "cat features, Conv1d→32, last step", early(trio))

    seq_v = models.TransformerSeq(35, 64)(vision)
    seq_a = models.TransformerSeq(74, 128)(audio)
    # Use a 64-d stand-in instead of 1024-d BERT encoder to keep the demo cheap.
    seq_t = models.TransformerSeq(768, 64)(text)
    late = models.LateFusionTransformer(in_dim=64 + 128 + 64, embed_dim=32)
    with torch.no_grad():
        _line(
            "LateFusionTransformer",
            "cat encoded seqs, Conv1d→32, last step",
            late([seq_v, seq_a, seq_t]),
        )

    class DemoParams:
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

    gmtm = models.GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=DemoParams)
    with torch.no_grad():
        _line("GMTM", "3-way cross-attn + gate + pool", gmtm(trio))

    pooled = models.AttentionPooling(768)(text)
    _line("AttentionPooling", "[B,T,768] weighted sum over T", pooled)

    print()
    print("Identity encoder leaves the stream unchanged; MultiFramework then")
    print("calls fusion(list_of_encoder_outputs) and finally head(fused).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
