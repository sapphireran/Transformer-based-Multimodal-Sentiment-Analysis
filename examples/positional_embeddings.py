"""Inspect sinusoidal positions the GMTM encoder adds to each modality."""

from __future__ import annotations

import torch

from examples.common import cpu_device, finite, print_kv


def table(seq_len: int = 8, dim: int = 16) -> torch.Tensor:
    from models import SinusoidalPositionalEmbedding

    emb = SinusoidalPositionalEmbedding(dim)
    # GMTM feeds the first feature channel (float), not integer token ids.
    # ``make_positions`` then masked-scatters into that tensor; a Long dummy
    # breaks on current PyTorch (Long vs Float).
    dummy = torch.arange(1, seq_len + 1, device=cpu_device(), dtype=torch.float).unsqueeze(0)
    return emb(dummy)  # [1, T, dim], detached


def main() -> int:
    pos = table()
    print("SinusoidalPositionalEmbedding (float dummy channel, detached)")
    print_kv(
        [
            ("shape", tuple(pos.shape)),
            ("requires_grad", pos.requires_grad),
            ("finite", finite(pos)),
            ("row 0 L2", float(pos[0, 0].norm())),
            ("row 1 L2", float(pos[0, 1].norm())),
            ("row 0 · row 1", float(torch.dot(pos[0, 0], pos[0, 1]))),
        ]
    )
    # First row is position 1 (index 0 is reserved for padding).
    if pos.shape != (1, 8, 16):
        raise RuntimeError(f"unexpected table shape {tuple(pos.shape)}")
    if pos.requires_grad:
        raise RuntimeError("positional table should be detached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
