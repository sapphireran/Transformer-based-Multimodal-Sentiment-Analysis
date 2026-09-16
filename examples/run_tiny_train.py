#!/usr/bin/env python3
"""A few CPU AdamW steps on ConcatLate and GMTM with synthetic clips.

This is not a substitute for ``train()``. It exists so you can confirm
that gradients flow through the real modules without MOSI/MOSEI,
CUDA, or ``memory_profiler``. Losses are printed; they should be
finite, and the last step is usually lower than the first on this
tiny random task.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))

from examples.repo import ensure_import_path

ensure_import_path()

import torch
import torch.nn as nn

from examples.shapes import BERT
from examples.synthetic_data import make_batch
from models import ConcatLate, GatedMultiTransfomerModel, Identity, LSTM, MLP


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


class ConcatLateBundle(nn.Module):
    """The ConcatLate triple from ``train_main_bert.py``, shrunk for CPU.

    Real sweep: LSTM 35/74/768 → 64/256/1024, then MLP(1344, 1344, 1).
    Demo:       LSTM 35/74/768 → 16/32/64,    then MLP(112, 64, 1).
    Packed lengths are not used; each LSTM sees a dense ``[B, T, F]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc_v = LSTM(35, 16, dropout=False, has_padding=False)
        self.enc_a = LSTM(74, 32, dropout=False, has_padding=False)
        self.enc_t = LSTM(768, 64, dropout=False, has_padding=False)
        self.fuse = ConcatLate()
        self.head = MLP(112, 64, 1)

    def forward(self, vision, audio, text):
        return self.head(self.fuse([self.enc_v(vision), self.enc_a(audio), self.enc_t(text)]))


class GMTMBundle(nn.Module):
    """Identity encoders + GMTM + Identity head, same wiring as the scripts."""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([Identity(), Identity(), Identity()])
        self.fuse = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=TinyHParams)
        self.head = Identity()

    def forward(self, vision, audio, text):
        encoded = [enc(x) for enc, x in zip(self.encoders, (vision, audio, text))]
        return self.head(self.fuse(encoded))


@dataclass
class TrainTrace:
    name: str
    losses: list[float]

    @property
    def first(self) -> float:
        return self.losses[0]

    @property
    def last(self) -> float:
        return self.losses[-1]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_module(module: nn.Module, steps: int, batch_size: int, seq_len: int, device: torch.device) -> TrainTrace:
    module.to(device)
    module.train()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.0)
    losses: list[float] = []
    for step in range(steps):
        batch = make_batch(BERT, batch_size=batch_size, seq_len=seq_len, seed=100 + step, device=device)
        pred = module(batch.vision, batch.audio, batch.text)
        if pred.shape != batch.labels.shape:
            pred = pred.view_as(batch.labels)
        loss = nn.functional.l1_loss(pred, batch.labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(module.parameters(), 8.0)
        opt.step()
        value = float(loss.item())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{module.__class__.__name__} produced a non-finite loss at step {step}")
        losses.append(value)
        print(f"  step {step + 1:>2}/{steps}  L1={value:.4f}")
    return TrainTrace(name=module.__class__.__name__, losses=losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    print(f"device={device}  steps={args.steps}  batch={args.batch_size}  T={args.seq_len}")

    torch.manual_seed(0)
    print("\n== ConcatLate bundle ==")
    late = train_module(ConcatLateBundle(), args.steps, args.batch_size, args.seq_len, device)
    print("\n== GMTM bundle ==")
    gmtm = train_module(GMTMBundle(), args.steps, args.batch_size, args.seq_len, device)

    print("\nsummary")
    for trace in (late, gmtm):
        delta = trace.last - trace.first
        print(
            f"  {trace.name:<18} first={trace.first:.4f}  "
            f"last={trace.last:.4f}  delta={delta:+.4f}"
        )
    print("Both bundles finished with finite L1.")


if __name__ == "__main__":
    main()
