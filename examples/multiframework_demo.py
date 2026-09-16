#!/usr/bin/env python3
"""Compose encoders → fusion → head the way ``train()`` wraps a sweep.

Two CPU graphs:

1. GMTM path from ``train_GMTM_bert.py`` — Identity encoders, GMTM fusion,
   Identity head.
2. Late-concat path from ``train_main_bert.py`` — mean-pool stand-ins for the
   LSTMs (so we stay packed-sequence-free on CPU), ConcatLate, MLP head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import BERT_DIM, dummy_modalities
from models import ConcatLate, GatedMultiTransfomerModel, Identity, MLP
from train_and_test import MultiFramework


class GmtmHParams:
    num_heads = 4
    layers = 2  # thinner than the recorded 4-layer run so the example stays snappy
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 32
    attn_mask = True
    output_dim = 1
    all_steps = False


class MeanPoolEncoder(nn.Module):
    """Cheap stand-in for the packed LSTM used in ConcatLate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run_gmtm(streams: list[torch.Tensor]) -> torch.Tensor:
    model = MultiFramework(
        encoders=[Identity(), Identity(), Identity()],
        fusion=GatedMultiTransfomerModel(3, [35, 74, BERT_DIM], hyp_params=GmtmHParams),
        head=Identity(),
    )
    model.eval()
    print(f"GMTM MultiFramework params: {_count(model):,}")
    with torch.no_grad():
        out = model(streams)
    print(f"  reps (encoder outs): {[tuple(r.shape) for r in model.reps]}")
    print(f"  fuseout:             {tuple(model.fuseout.shape)}")
    print(f"  head output:         {tuple(out.shape)}  {out.squeeze(1).tolist()}")
    return out


def run_late_concat(streams: list[torch.Tensor]) -> torch.Tensor:
    # Mean-pool each stream to a vector, then ConcatLate + MLP(35+74+768 → 1).
    fused_dim = 35 + 74 + BERT_DIM
    model = MultiFramework(
        encoders=[MeanPoolEncoder(), MeanPoolEncoder(), MeanPoolEncoder()],
        fusion=ConcatLate(),
        head=MLP(fused_dim, fused_dim, 1),
    )
    model.eval()
    print(f"\nLate-concat MultiFramework params: {_count(model):,}")
    with torch.no_grad():
        out = model(streams)
    print(f"  reps:        {[tuple(r.shape) for r in model.reps]}")
    print(f"  fuseout:     {tuple(model.fuseout.shape)}  (should be B x {fused_dim})")
    print(f"  head output: {tuple(out.shape)}  {out.squeeze(1).tolist()}")
    return out


def main() -> None:
    torch.manual_seed(0)
    visual, audio, text, labels = dummy_modalities(text_dim=BERT_DIM)
    streams = [visual, audio, text]
    print(f"Batch streams: {[tuple(s.shape) for s in streams]}  labels={labels.squeeze(1).tolist()}\n")
    gmtm_out = run_gmtm(streams)
    late_out = run_late_concat(streams)
    criterion = nn.L1Loss()
    print(f"\nUntrained L1  GMTM={criterion(gmtm_out, labels).item():.4f}  "
          f"late-concat={criterion(late_out, labels).item():.4f}")
    print("Both graphs are the same wrapper ``train()`` optimizes; only the modules change.")


if __name__ == "__main__":
    main()
