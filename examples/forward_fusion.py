#!/usr/bin/env python3
"""CPU forward pass for every fusion graph used in the bake-off scripts.

No MOSI/MOSEI pickle is required. Tensors use the BERT feature widths
(35 / 74 / 768) and a short sequence so this finishes in a few seconds
on a laptop.

Run from the repository root:

    python examples/forward_fusion.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._bootstrap import ensure_output_dir

import torch
import torch.nn as nn

from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    GRUWithLinear,
    Identity,
    LateFusionTransformer,
    LowRankTensorFusion,
    LSTM,
    MLP,
    TensorFusion,
    TransformerSeq,
)
from synthetic import make_synthetic_batch


class HParams:
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


def _run(name: str, fn) -> tuple[str, tuple[int, ...]]:
    torch.manual_seed(0)
    out = fn()
    if not torch.is_tensor(out):
        raise TypeError(f"{name} returned {type(out)!r}, expected a tensor")
    if not torch.isfinite(out).all():
        raise RuntimeError(f"{name} produced non-finite values")
    return name, tuple(out.shape)


def concat_early(batch):
    fused = ConcatEarly()(batch.as_list())
    # Training uses a packed LSTM head; here we collapse time with mean.
    pooled = fused.mean(dim=1)
    return MLP(pooled.size(-1), 64, 1)(pooled)


def concat_late(batch):
    encoders = [
        LSTM(35, 32, dropout=False, has_padding=False),
        LSTM(74, 32, dropout=False, has_padding=False),
        LSTM(768, 64, dropout=False, has_padding=False),
    ]
    encoded = [enc(x) for enc, x in zip(encoders, batch.as_list())]
    fused = ConcatLate()(encoded)
    return MLP(fused.size(-1), 64, 1)(fused)


def low_rank(batch):
    encoders = [
        GRUWithLinear(35, 32, 16, dropout=False, has_padding=True),
        GRUWithLinear(74, 32, 16, dropout=False, has_padding=True),
        GRUWithLinear(768, 64, 32, dropout=False, has_padding=True),
    ]
    encoded = [
        enc([x, batch.lengths]) for enc, x in zip(encoders, batch.as_list())
    ]
    fused = LowRankTensorFusion([16, 16, 32], 32, rank=4)(encoded)
    return MLP(32, 32, 1)(fused)


def tensor_fusion(batch):
    encoders = [
        GRUWithLinear(35, 16, 8, dropout=False, has_padding=True),
        GRUWithLinear(74, 16, 8, dropout=False, has_padding=True),
        GRUWithLinear(768, 32, 8, dropout=False, has_padding=True),
    ]
    encoded = [
        enc([x, batch.lengths]) for enc, x in zip(encoders, batch.as_list())
    ]
    fused = TensorFusion()(encoded)
    return MLP(fused.size(-1), 64, 1)(fused)


def transformer_early(batch):
    fused = EarlyFusionTransformer(n_features=batch.total_feature_dim())(batch.as_list())
    # The bake-off script attaches MLP(64, 64, 1), but the module emits embed_dim=32.
    return MLP(fused.size(-1), fused.size(-1), 1)(fused)


def transformer_late(batch):
    encoders = [
        TransformerSeq(35, 16),
        TransformerSeq(74, 16),
        TransformerSeq(768, 32),
    ]
    encoded = [enc(x) for enc, x in zip(encoders, batch.as_list())]
    in_dim = sum(t.size(-1) for t in encoded)
    fused = LateFusionTransformer(in_dim=in_dim, embed_dim=32)(encoded)
    return MLP(fused.size(-1), 32, 1)(fused)


def gmtm(batch):
    fusion = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=HParams)
    # Identity encoders / head, matching train_GMTM_bert.py
    return fusion(batch.as_list())


def main() -> None:
    batch = make_synthetic_batch(
        batch_size=4, seq_len=12, embedding="bert", seed=0, include_signal=True
    )
    jobs = [
        ("ConcatEarly", lambda: concat_early(batch)),
        ("ConcatLate", lambda: concat_late(batch)),
        ("LowRankTensorFusion", lambda: low_rank(batch)),
        ("TensorFusion", lambda: tensor_fusion(batch)),
        ("TransformerEarly", lambda: transformer_early(batch)),
        ("TransformerLate", lambda: transformer_late(batch)),
        ("GatedMultiTransfomer", lambda: gmtm(batch)),
    ]

    print(f"synthetic batch  B={batch.visual.size(0)}  T={batch.visual.size(1)}")
    print(
        "widths          "
        f"visual={batch.visual.size(-1)}  "
        f"audio={batch.audio.size(-1)}  "
        f"text={batch.text.size(-1)}"
    )
    print()

    lines = ["method,out_shape"]
    for name, fn in jobs:
        label, shape = _run(name, fn)
        print(f"{label:<24} -> {shape}")
        lines.append(f"{label},\"{shape}\"")

    out_dir = ensure_output_dir()
    path = out_dir / "forward_fusion_shapes.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    # TransformerEncoderLayer warns on nested tensors; keep the demo quiet.
    import warnings

    warnings.filterwarnings("ignore", message=".*nested tensors.*")
    torch.set_grad_enabled(False)
    nn.TransformerEncoderLayer  # keep import style obvious for readers
    main()
