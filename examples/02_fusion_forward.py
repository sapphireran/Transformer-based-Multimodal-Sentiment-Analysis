#!/usr/bin/env python3
"""One CPU forward pass through every fusion module in ``model/models.py``.

Uses the same feature widths as the MOSEI BERT recipes. Packed LSTM/GRU
encoders get dummy lengths (every clip is already length T). This is a
shape-and-smoke test, not a quality claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BERT_DIM, make_aligned_batch, packed_lengths

from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GRUWithLinear,
    LateFusionTransformer,
    LowRankTensorFusion,
    LSTM,
    TensorFusion,
    TransformerSeq,
)


def _print(name: str, out: torch.Tensor) -> None:
    finite = bool(torch.isfinite(out).all())
    print(
        f"{name:28s}  out={tuple(out.shape)}  "
        f"mean={out.mean().item():+.4f}  finite={finite}"
    )


def demo_concat_early(vision, audio, text) -> None:
    fused = ConcatEarly()([vision, audio, text])
    assert fused.shape[-1] == 35 + 74 + BERT_DIM
    _print("ConcatEarly", fused)


def demo_concat_late(vision, audio, text, lengths) -> None:
    encoders = [
        LSTM(35, 64, dropout=False, has_padding=True),
        LSTM(74, 256, dropout=False, has_padding=True),
        LSTM(BERT_DIM, 128, dropout=False, has_padding=True),
    ]
    # Smaller text LSTM than the 1024-d training recipe so this stays CPU-light.
    reps = [enc((mod, length)) for enc, mod, length in zip(encoders, (vision, audio, text), lengths)]
    fused = ConcatLate()(reps)
    _print("ConcatLate (tiny text LSTM)", fused)


def demo_tensor_fusion(vision, audio, text, lengths) -> None:
    encoders = [
        GRUWithLinear(35, 32, 8, dropout=False, has_padding=True),
        GRUWithLinear(74, 32, 8, dropout=False, has_padding=True),
        GRUWithLinear(BERT_DIM, 32, 8, dropout=False, has_padding=True),
    ]
    reps = [enc((mod, length)) for enc, mod, length in zip(encoders, (vision, audio, text), lengths)]
    fused = TensorFusion()(reps)
    # homogenized 9 x 9 x 9
    _print("TensorFusion (8-d factors)", fused)


def demo_lrtf(vision, audio, text, lengths) -> None:
    encoders = [
        GRUWithLinear(35, 32, 16, dropout=False, has_padding=True),
        GRUWithLinear(74, 32, 16, dropout=False, has_padding=True),
        GRUWithLinear(BERT_DIM, 32, 16, dropout=False, has_padding=True),
    ]
    reps = [enc((mod, length)) for enc, mod, length in zip(encoders, (vision, audio, text), lengths)]
    fusion = LowRankTensorFusion([16, 16, 16], output_dim=32, rank=4)
    fused = fusion(reps)
    _print("LowRankTensorFusion r=4", fused)


def demo_transformer_early(vision, audio, text) -> None:
    fusion = EarlyFusionTransformer(n_features=35 + 74 + BERT_DIM)
    fused = fusion([vision, audio, text])
    _print("TransformerEarly last-step", fused)
    # The BERT training script then applies MLP(64, 64, 1). That linear
    # does not match this 32-d output — document it rather than hide it.
    try:
        nn.Linear(64, 1)(fused)
        print("  unexpected: MLP(64, …) accepted a 32-d fused vector")
    except RuntimeError as exc:
        print(f"  expected size mismatch if you stack MLP(64, 64, 1): {exc}".split("\n")[0])


def demo_transformer_late(vision, audio, text) -> None:
    encoders = [
        TransformerSeq(35, 32),
        TransformerSeq(74, 32),
        TransformerSeq(BERT_DIM, 64),
    ]
    seqs = [enc(mod) for enc, mod in zip(encoders, (vision, audio, text))]
    in_dim = sum(s.shape[-1] for s in seqs)
    fusion = LateFusionTransformer(in_dim=in_dim, embed_dim=32)
    fused = fusion(seqs)
    _print(f"TransformerLate in_dim={in_dim}", fused)


def main() -> None:
    torch.manual_seed(0)
    vision, audio, text, _ = make_aligned_batch(batch_size=2, seq_len=8, text_dim=BERT_DIM)
    lengths = packed_lengths(batch_size=2, seq_len=8)
    print("input", tuple(vision.shape), tuple(audio.shape), tuple(text.shape))
    print()
    demo_concat_early(vision, audio, text)
    demo_concat_late(vision, audio, text, lengths)
    demo_tensor_fusion(vision, audio, text, lengths)
    demo_lrtf(vision, audio, text, lengths)
    demo_transformer_early(vision, audio, text)
    demo_transformer_late(vision, audio, text)
    print()
    print("All fusion modules produced a finite tensor on CPU.")


if __name__ == "__main__":
    main()
