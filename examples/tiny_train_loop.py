"""Two-epoch CPU stand-in for the MultiFramework + GMTM + L1 + AdamW recipe."""

from __future__ import annotations

import torch

from examples.common import TinyGMTMParams, cpu_device, print_kv
from examples.forward_gmtm import build_gmtm
from examples.synthetic_data import make_batch


def train_toy(
    steps_per_epoch: int = 6,
    epochs: int = 2,
    batch_size: int = 6,
    seq_len: int = 8,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    device = cpu_device()
    model = build_gmtm(hyp_params=TinyGMTMParams).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.L1Loss()
    history: dict[str, list[float]] = {"train": [], "valid": []}

    # One fixed validation batch; train batches rotate the seed.
    valid = make_batch(batch_size=batch_size, seq_len=seq_len, seed=99)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step in range(steps_per_epoch):
            batch = make_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                seed=1000 + epoch * 50 + step,
            )
            opt.zero_grad()
            pred = model([t.to(device) for t in batch.as_list()])
            loss = loss_fn(pred, batch.labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            opt.step()
            running += float(loss.item())
        train_mae = running / steps_per_epoch
        history["train"].append(train_mae)

        model.eval()
        with torch.no_grad():
            v_pred = model([t.to(device) for t in valid.as_list()])
            v_loss = float(loss_fn(v_pred, valid.labels.to(device)).item())
        history["valid"].append(v_loss)
    return history


def main() -> int:
    history = train_toy()
    print("Toy GMTM fit (Identity-style: fusion owns the graph, L1, AdamW, clip=8)")
    for epoch, (tr, va) in enumerate(zip(history["train"], history["valid"]), start=1):
        print(f"  epoch {epoch}: train L1={tr:.4f}  valid L1={va:.4f}")
    print_kv(
        [
            ("epochs", len(history["train"])),
            ("first train L1", f"{history['train'][0]:.4f}"),
            ("last train L1", f"{history['train'][-1]:.4f}"),
        ]
    )
    if any(v != v for v in history["train"] + history["valid"]):
        raise RuntimeError("NaN in toy training history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
