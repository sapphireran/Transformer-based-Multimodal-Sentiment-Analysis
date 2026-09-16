"""Show AttentionPooling weights summing to 1 along time."""

from __future__ import annotations

import torch

from examples.common import cpu_device, finite, print_kv
from examples.synthetic_data import make_batch


def pool_and_weights(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from models import AttentionPooling

    pool = AttentionPooling(x.size(-1))
    weights = torch.softmax(pool.attention(x).squeeze(-1), dim=1)
    out = pool(x)
    return out, weights


def main() -> int:
    device = cpu_device()
    batch = make_batch(batch_size=2, seq_len=8)
    x = torch.cat([batch.vision, batch.audio[:, :, :16]], dim=-1).to(device)
    out, weights = pool_and_weights(x)
    print("AttentionPooling on a concatenated [B, T, F] toy sequence")
    print_kv(
        [
            ("input", tuple(x.shape)),
            ("pooled", tuple(out.shape)),
            ("weights", tuple(weights.shape)),
            ("row sums", weights.sum(dim=1).tolist()),
            ("finite", finite(out) and finite(weights)),
        ]
    )
    if not torch.allclose(weights.sum(dim=1), torch.ones(x.size(0)), atol=1e-5):
        raise RuntimeError("attention weights must sum to 1 over T")
    if out.shape != (x.size(0), x.size(-1)):
        raise RuntimeError(f"unexpected pooled shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
