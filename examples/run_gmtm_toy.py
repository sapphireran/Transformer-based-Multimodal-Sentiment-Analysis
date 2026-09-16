#!/usr/bin/env python3
"""Train GMTM for a few CPU epochs on synthetic affect batches.

The label is a known function of the text stream (see synthetic_affect.py),
so a 16-d, 1-layer GMTM should drive L1 down. This is a smoke test that the
module's cross-attention / gate / pool / head path is differentiable — not
a published score.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW

from device import describe_device, get_device
from metrics_lib import evaluate_affect_batch, format_metrics
from paths import ensure_model_on_path, ensure_output_dir
from synthetic_affect import (
    AffectShapes,
    iter_maxpad_batches,
    make_pickle_dict,
    zero_unused_modalities,
)

ensure_model_on_path()
from models import GatedMultiTransfomerModel  # noqa: E402


class TinyHParams:
    """Small legal (embed_dim % num_heads == 0) config for a laptop CPU."""

    num_heads = 2
    layers = 1
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 16
    attn_mask = True
    output_dim = 1
    all_steps = False


def _epoch(
    model: nn.Module,
    split: dict,
    batch_size: int,
    device: torch.device,
    optimizer: AdamW | None,
    keep: List[str],
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    loss_fn = nn.L1Loss()
    for vision, audio, text, labels in iter_maxpad_batches(split, batch_size, device=device):
        vision, audio, text = zero_unused_modalities(vision, audio, text, keep)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model([vision, audio, text])
        loss = loss_fn(pred, labels)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
        total += float(loss.item()) * labels.size(0)
        count += labels.size(0)
    return total / max(count, 1)


@torch.no_grad()
def _predict(model: nn.Module, split: dict, batch_size: int, device: torch.device, keep: List[str]):
    model.eval()
    preds = []
    trues = []
    for vision, audio, text, labels in iter_maxpad_batches(split, batch_size, device=device):
        vision, audio, text = zero_unused_modalities(vision, audio, text, keep)
        preds.append(model([vision, audio, text]).detach().cpu())
        trues.append(labels.detach().cpu())
    return torch.cat(trues, 0).numpy(), torch.cat(preds, 0).numpy()


def run(
    epochs: int = 6,
    batch_size: int = 16,
    n_train: int = 96,
    n_valid: int = 32,
    seq_len: int = 20,
    seed: int = 0,
    keep: List[str] | None = None,
) -> Dict:
    keep = keep or ["text", "audio", "visual"]
    device = get_device()
    torch.manual_seed(seed)
    shapes = AffectShapes.for_embedding("bert", max_len=seq_len)
    data = make_pickle_dict(n_train=n_train, n_valid=n_valid, n_test=n_valid, shapes=shapes, seed=seed)
    model = GatedMultiTransfomerModel(3, [shapes.visual, shapes.audio, shapes.text], hyp_params=TinyHParams).to(
        device
    )
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0.01)

    history = []
    print(f"device={describe_device(device)}  keep={keep}  params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, data["train"], batch_size, device, optimizer, keep)
        valid_loss = _epoch(model, data["valid"], batch_size, device, None, keep)
        history.append({"epoch": epoch, "train_l1": train_loss, "valid_l1": valid_loss})
        print(f"epoch {epoch:02d}  train_l1={train_loss:.4f}  valid_l1={valid_loss:.4f}")

    y, yhat = _predict(model, data["valid"], batch_size, device, keep)
    metrics = evaluate_affect_batch(y, yhat)
    print("valid", format_metrics(metrics))

    dropped = history[0]["train_l1"] - history[-1]["train_l1"]
    ok = history[-1]["train_l1"] < history[0]["train_l1"]
    report = {
        "device": describe_device(device),
        "keep": keep,
        "embed_dim": TinyHParams.embed_dim,
        "layers": TinyHParams.layers,
        "history": history,
        "valid_metrics": metrics,
        "train_l1_drop": dropped,
        "ok": ok,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep",
        nargs="+",
        default=["text", "audio", "visual"],
        help="modalities to keep; others are zeroed (ablation protocol)",
    )
    args = parser.parse_args()
    report = run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_train=args.n_train,
        seq_len=args.seq_len,
        seed=args.seed,
        keep=args.keep,
    )
    out = ensure_output_dir() / "gmtm_toy.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    if not report["ok"]:
        print("train L1 did not decrease; failing the smoke test")
        return 1
    print("smoke test passed: train L1 decreased (valid L1 is not part of the gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
