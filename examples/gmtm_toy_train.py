#!/usr/bin/env python3
"""Short L1 training loop on synthetic clips.

Labels are a noisy function of the text mean (see ``synthetic_data.make_split``),
so a small GMTM should drive training MAE down without MOSI / MOSEI files.
This is a smoke test for the optimizer path, not a published result.
"""

from __future__ import annotations

import json

from synthetic_data import (
    FeatureSpec,
    ensure_model_on_path,
    make_dataset,
    pick_device,
    seed_everything,
    tensors_from_split,
)

ensure_model_on_path()

import torch
from models import GatedMultiTransfomerModel
from train_and_test import eval_affect, split_uniform_5, split_uniform_7
from sklearn.metrics import accuracy_score


class ToyHParams:
    num_heads = 4
    layers = 2
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


def batches(split, batch_size: int, device, shuffle: bool):
    vision, audio, text, labels = tensors_from_split(split, device)
    n = labels.size(0)
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield [vision[idx], audio[idx], text[idx]], labels[idx]


def run(steps: int = 25, batch_size: int = 8, lr: float = 3e-3) -> dict:
    seed_everything(21)
    spec = FeatureSpec()
    device = pick_device()
    data = make_dataset(n_train=48, n_valid=16, n_test=16, spec=spec, seed=21)
    model = GatedMultiTransfomerModel(3, spec.as_list, hyp_params=ToyHParams).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.L1Loss()

    train_losses = []
    model.train()
    step = 0
    while step < steps:
        for streams, labels in batches(data["train"], batch_size, device, shuffle=True):
            opt.zero_grad(set_to_none=True)
            pred = model(streams)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            opt.step()
            train_losses.append(float(loss.item()))
            step += 1
            if step >= steps:
                break

    model.eval()
    with torch.no_grad():
        streams, labels = next(batches(data["test"], 16, device, shuffle=False))
        pred = model(streams)
        test_mae = float(loss_fn(pred, labels).item())
        y = labels.detach().cpu().numpy()
        yhat = pred.detach().cpu().numpy()
        acc7 = float(accuracy_score(split_uniform_7(y), split_uniform_7(yhat)))
        acc5 = float(accuracy_score(split_uniform_5(y), split_uniform_5(yhat)))
        f1, acc2 = eval_affect(y, yhat)

    first = sum(train_losses[:5]) / max(len(train_losses[:5]), 1)
    last = sum(train_losses[-5:]) / max(len(train_losses[-5:]), 1)
    return {
        "device": str(device),
        "steps": steps,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_mae_first5": round(first, 4),
        "train_mae_last5": round(last, 4),
        "loss_went_down": last < first,
        "test_mae": round(test_mae, 4),
        "test_acc7": round(acc7, 4),
        "test_acc5": round(acc5, 4),
        "test_acc2": round(float(acc2), 4),
        "test_f1": round(float(f1), 4),
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
