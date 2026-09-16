"""Overfit a tiny GMTM to synthetic labels for a few Adam steps.

Not a real experiment — just proof that the module is differentiable
on CPU and that L1 on a 16-sample toy set can move.
"""

from __future__ import annotations

import torch

import _paths  # noqa: F401
from gmtm_forward import TinyHParams, build_tiny_gmtm
from synthetic_batch import make_synthetic_batch


def run_overfit(
    steps: int = 25,
    batch_size: int = 16,
    seq_len: int = 8,
    lr: float = 5e-3,
    seed: int = 1,
) -> list[float]:
    torch.manual_seed(seed)
    n_features = [8, 10, 12]
    model = build_tiny_gmtm(n_features)
    model.train()
    batch = make_synthetic_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        visual_dim=8,
        audio_dim=10,
        text_dim=12,
        seed=seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    history: list[float] = []
    for step in range(steps):
        opt.zero_grad()
        pred = model(batch.as_gmtm_input())
        loss = loss_fn(pred, batch.labels)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history


def _demo() -> list[float]:
    history = run_overfit()
    print("tiny GMTM overfit (synthetic L1)")
    print(f"  step  1  {history[0]:.4f}")
    print(f"  step {len(history):>2}  {history[-1]:.4f}")
    dropped = history[0] - history[-1]
    print(f"  drop     {dropped:.4f}")
    if dropped <= 0:
        raise SystemExit("loss did not decrease; check the GMTM forward path")
    return history


if __name__ == "__main__":
    _demo()
