#!/usr/bin/env python3
"""Forward every fusion operator on a synthetic MOSI/MOSEI-shaped batch.

This is the fastest way to see tensor shapes without downloading the
corpora or loading a ``.pt`` file. Sequence-level methods
(``ConcatEarly``, the two Transformer fusions) consume ``[B, T, F]``
streams. Vector-level methods (late concat, both tensor fusions)
consume already-pooled ``[B, D]`` vectors, which is what the LSTM /
GRU encoders emit in ``train_main_*.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))

from examples.repo import ensure_import_path

ensure_import_path()

import torch

from examples.shapes import BERT, GLOVE, FeatureSpec
from examples.synthetic_data import make_batch, make_vector_batch
from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _show(name: str, tensor: torch.Tensor) -> None:
    finite = bool(torch.isfinite(tensor).all())
    print(
        f"  {name:<24} shape={tuple(tensor.shape)}  "
        f"finite={finite}  mean={tensor.float().mean().item():+.4f}"
    )


def run_sequence_fusions(spec: FeatureSpec, batch_size: int, seq_len: int, device: torch.device) -> None:
    batch = make_batch(spec, batch_size=batch_size, seq_len=seq_len, seed=1, device=device)
    streams = batch.as_list()
    print(f"\n== sequence fusions ({spec.early_width}-d early width) ==")
    print(f"   inputs: {[tuple(x.shape) for x in streams]}")

    early = ConcatEarly().to(device)
    _show("ConcatEarly", early(streams))

    early_tf = EarlyFusionTransformer(n_features=spec.early_width).to(device)
    _show("TransformerEarly", early_tf(streams))

    # LateFusionTransformer concatenates on the last dim, so the three
    # streams must already share a time axis — the raw aligned features
    # do. The Conv1d width is the concatenated feature size.
    late_tf = LateFusionTransformer(in_dim=spec.early_width, embed_dim=32).to(device)
    _show("TransformerLate(raw)", late_tf(streams))


def run_vector_fusions(spec: FeatureSpec, batch_size: int, device: torch.device) -> None:
    # Hidden sizes copied from train_main_bert.py / train_main_glove.py
    # so the fused widths match the MLP heads in those scripts.
    if spec.text == BERT.text:
        late_dims = (64, 256, 1024)
        lrtf_dims = (32, 64, 256)
        lrtf_out = 256
        tfn_dims = (19, 39, 159)
    else:
        late_dims = (64, 256, 512)
        lrtf_dims = (32, 64, 128)
        lrtf_out = 128
        tfn_dims = (19, 39, 79)

    print(f"\n== vector fusions (text dim {spec.text}) ==")

    late = ConcatLate().to(device)
    late_in = make_vector_batch(late_dims, batch_size=batch_size, seed=2, device=device)
    print(f"   ConcatLate inputs: {[tuple(x.shape) for x in late_in]}")
    _show("ConcatLate", late(late_in))

    tfn = TensorFusion().to(device)
    tfn_in = make_vector_batch(tfn_dims, batch_size=batch_size, seed=3, device=device)
    print(f"   TensorFusion inputs: {[tuple(x.shape) for x in tfn_in]}")
    fused = tfn(tfn_in)
    expected = 1
    for d in tfn_dims:
        expected *= d + 1
    _show("TensorFusion", fused)
    print(f"   expected fused width (product of d_i+1): {expected}")

    lrtf = LowRankTensorFusion(list(lrtf_dims), lrtf_out, rank=32).to(device)
    lrtf_in = make_vector_batch(lrtf_dims, batch_size=batch_size, seed=4, device=device)
    print(f"   LowRankTensorFusion inputs: {[tuple(x.shape) for x in lrtf_in]}")
    _show("LowRankTensorFusion", lrtf(lrtf_in))

    # TransformerFusion wants a shared d_model. Project the three late
    # vectors to 64-d first so the demo stays close to "modality tokens".
    shared = 64
    proj = [torch.nn.Linear(d, shared).to(device) for d in late_dims]
    tok_in = [p(x) for p, x in zip(proj, late_in)]
    tfuse = TransformerFusion(d_model=shared, nhead=4, num_layers=2).to(device)
    print(f"   TransformerFusion inputs: {[tuple(x.shape) for x in tok_in]}")
    _show("TransformerFusion", tfuse(tok_in))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=("bert", "glove"), default="bert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BERT if args.embedding == "bert" else GLOVE
    device = _device(args.device)
    print(f"device={device}  embedding={args.embedding}  spec={spec}")
    torch.manual_seed(0)
    run_sequence_fusions(spec, args.batch_size, args.seq_len, device)
    run_vector_fusions(spec, args.batch_size, device)
    print("\nAll fusion operators produced a finite tensor.")


if __name__ == "__main__":
    main()
