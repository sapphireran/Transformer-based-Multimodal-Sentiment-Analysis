"""Print tensor ranks for every fusion module in ``models.py``.

Toy widths, CPU, no dataloaders. This is the fastest way to see why
``TensorFusion`` needs a 128k-D head on the BERT study (outer product
of lifted unimodal vectors) and why late concat is just a cat of
already-encoded vectors.
"""

from __future__ import annotations

import torch

import _paths  # noqa: F401
from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
)


def _banner(name: str, extra: str = "") -> None:
    print(f"{name}{': ' + extra if extra else ''}")


def demo_concat(batch_size: int = 3, seq_len: int = 5) -> None:
    vision = torch.randn(batch_size, seq_len, 8)
    audio = torch.randn(batch_size, seq_len, 10)
    text = torch.randn(batch_size, seq_len, 12)
    early = ConcatEarly()([vision, audio, text])
    late = ConcatLate()([vision, audio, text])
    _banner("ConcatEarly", "cat on dim=2 (channel axis)")
    print(f"  in  vis{tuple(vision.shape)} aud{tuple(audio.shape)} "
          f"txt{tuple(text.shape)}")
    print(f"  out {tuple(early.shape)}  (8+10+12=30 channels)")
    _banner("ConcatLate", "flatten each stream, cat on dim=1")
    print(f"  out {tuple(late.shape)}  (5*8 + 5*10 + 5*12 = 150)")


def demo_tensor_fusion(batch_size: int = 3) -> None:
    # Same lifted sizes as train_main_bert.py: 19, 39, 159 → 20*40*160.
    a = torch.randn(batch_size, 19)
    b = torch.randn(batch_size, 39)
    c = torch.randn(batch_size, 159)
    out = TensorFusion()([a, b, c])
    _banner("TensorFusion", "outer product after prepending 1")
    print(f"  in  {tuple(a.shape)} {tuple(b.shape)} {tuple(c.shape)}")
    print(f"  out {tuple(out.shape)}  expected last dim 20*40*160=128000")


def demo_lrtf(batch_size: int = 3) -> None:
    dims = [32, 64, 128]
    mods = [torch.randn(batch_size, d) for d in dims]
    out = LowRankTensorFusion(dims, output_dim=16, rank=4)(mods)
    _banner("LowRankTensorFusion", "rank-4 factorization, out_dim=16")
    print(f"  in  {[tuple(m.shape) for m in mods]}")
    print(f"  out {tuple(out.shape)}")


def demo_transformer_fusion(batch_size: int = 3) -> None:
    d_model = 16
    mods = [torch.randn(batch_size, d_model) for _ in range(3)]
    out = TransformerFusion(d_model=d_model, nhead=4, num_layers=1)(mods)
    _banner("TransformerFusion", "stack modalities as tokens, mean-pool")
    print(f"  in  3 x {tuple(mods[0].shape)}")
    print(f"  out {tuple(out.shape)}")


def demo_early_late_transformers(batch_size: int = 2, seq_len: int = 6) -> None:
    n_features = 8 + 10 + 12
    vision = torch.randn(batch_size, seq_len, 8)
    audio = torch.randn(batch_size, seq_len, 10)
    text = torch.randn(batch_size, seq_len, 12)
    early = EarlyFusionTransformer(n_features=n_features)
    # Override embed_dim access: the class attr is 32.
    late = LateFusionTransformer(in_dim=n_features, embed_dim=32)
    early_out = early([vision, audio, text])
    late_out = late([vision, audio, text])
    _banner("EarlyFusionTransformer", "1x1 conv → 4-layer encoder → last step")
    print(f"  out {tuple(early_out.shape)}  (embed_dim={early.embed_dim})")
    _banner("LateFusionTransformer", "same, after a feature-axis cat")
    print(f"  out {tuple(late_out.shape)}  (embed_dim={late.embed_dim})")


def _demo() -> None:
    torch.manual_seed(0)
    demo_concat()
    demo_tensor_fusion()
    demo_lrtf()
    demo_transformer_fusion()
    demo_early_late_transformers()


if __name__ == "__main__":
    _demo()
