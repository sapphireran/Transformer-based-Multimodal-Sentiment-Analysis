#!/usr/bin/env python3
"""Print input/output ranks for every fusion module used in the MOSEI sweeps."""

from __future__ import annotations

import torch

from common import BERT_DIM, dummy_modalities
from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
)


def _line(name: str, inputs: list[torch.Tensor], output: torch.Tensor) -> None:
    in_shapes = [tuple(t.shape) for t in inputs]
    print(f"{name:24} {in_shapes}  ->  {tuple(output.shape)}")


def main() -> None:
    torch.manual_seed(0)
    visual, audio, text, _ = dummy_modalities(text_dim=BERT_DIM)
    batch = visual.size(0)

    print("Sequence-level fusion (early / transformer-over-time)\n")
    early = ConcatEarly()
    _line("ConcatEarly", [visual, audio, text], early([visual, audio, text]))

    eft = EarlyFusionTransformer(n_features=35 + 74 + BERT_DIM)
    eft.eval()
    with torch.no_grad():
        _line("EarlyFusionTransformer", [visual, audio, text], eft([visual, audio, text]))

    # Late transformer sees per-modality encoder outputs already projected.
    enc_v = torch.randn(batch, visual.size(1), 64)
    enc_a = torch.randn(batch, visual.size(1), 128)
    enc_t = torch.randn(batch, visual.size(1), 1024)
    lft = LateFusionTransformer(in_dim=64 + 128 + 1024)
    lft.eval()
    with torch.no_grad():
        _line("LateFusionTransformer", [enc_v, enc_a, enc_t], lft([enc_v, enc_a, enc_t]))

    print("\nVector-level fusion (after an LSTM/GRU encoder collapses time)\n")
    vecs = [
        torch.randn(batch, 64),
        torch.randn(batch, 256),
        torch.randn(batch, 1024),
    ]
    _line("ConcatLate", vecs, ConcatLate()(vecs))

    tfn_in = [torch.randn(batch, 19), torch.randn(batch, 39), torch.randn(batch, 159)]
    tfn_out = TensorFusion()(tfn_in)
    _line("TensorFusion BERT", tfn_in, tfn_out)
    expected = (19 + 1) * (39 + 1) * (159 + 1)
    print(f"{'':24} expected flattened width {expected} (outer product + ones)")

    lmf = LowRankTensorFusion([32, 64, 256], output_dim=256, rank=32)
    lmf.eval()
    lmf_in = [torch.randn(batch, 32), torch.randn(batch, 64), torch.randn(batch, 256)]
    with torch.no_grad():
        _line("LowRankTensorFusion", lmf_in, lmf(lmf_in))

    same_dim = [torch.randn(batch, 64) for _ in range(3)]
    tfuse = TransformerFusion(d_model=64, nhead=4, num_layers=2)
    tfuse.eval()
    with torch.no_grad():
        _line("TransformerFusion", same_dim, tfuse(same_dim))

    print(
        "\nReading the ranks:\n"
        "  ConcatEarly keeps time and grows the feature axis (877 = 35+74+768).\n"
        "  EarlyFusionTransformer pools the last time step to a 32-d clip vector.\n"
        "  LateFusionTransformer does the same after per-stream transformers.\n"
        "  ConcatLate / LMF / TFN expect time to be gone already.\n"
        "  TFN explodes to 128000-d; LMF keeps a chosen output_dim (256 here)."
    )


if __name__ == "__main__":
    main()
