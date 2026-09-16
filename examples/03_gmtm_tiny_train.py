#!/usr/bin/env python3
"""Train a tiny GMTM on synthetic MOSI labels for a few CPU steps.

Mirrors ``train_GMTM_bert.py``: Identity encoders, GMTM fusion, Identity head,
L1 loss. Reports the same metric keys as the real trainers.

    python examples/03_gmtm_tiny_train.py
    python examples/03_gmtm_tiny_train.py --steps 30 --batch-size 8
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from common import (
    TINY_AUDIO_DIM,
    TINY_TEXT_DIM,
    TINY_VISION_DIM,
    TinyHParams,
    batch_from_split,
    count_parameters,
    device,
    make_synthetic_mosi,
)
from metrics import compute_affect_metrics, format_metrics_row
from models import GatedMultiTransfomerModel, Identity
from train_and_test import MultiFramework


def iterate_minibatches(modalities, labels, batch_size: int):
    n = labels.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield [m[start:end] for m in modalities], labels[start:end]


def evaluate(model: nn.Module, modalities, labels) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(modalities)
    return compute_affect_metrics(labels, pred, plot_confusion=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    data = make_synthetic_mosi(n_train=32, n_valid=8, n_test=8, seed=args.seed)
    train_x, train_y = batch_from_split(data["train"], dev)
    valid_x, valid_y = batch_from_split(data["valid"], dev)
    test_x, test_y = batch_from_split(data["test"], dev)

    encoders = [Identity(), Identity(), Identity()]
    fusion = GatedMultiTransfomerModel(
        3,
        [TINY_VISION_DIM, TINY_AUDIO_DIM, TINY_TEXT_DIM],
        hyp_params=TinyHParams,
    )
    head = Identity()
    model = MultiFramework(encoders, fusion, head, has_padding=False).to(dev)
    criterion = nn.L1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(
        f"device={dev}  params={count_parameters(model)}  "
        f"train={train_y.size(0)} valid={valid_y.size(0)} test={test_y.size(0)}"
    )

    best_valid = float("inf")
    best_state = None
    step = 0
    while step < args.steps:
        model.train()
        for xb, yb in iterate_minibatches(train_x, train_y, args.batch_size):
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optim.step()
            step += 1
            if step >= args.steps:
                break
        valid = evaluate(model, valid_x, valid_y)
        marker = ""
        if valid["MAE"] < best_valid:
            best_valid = valid["MAE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best valid MAE"
        print(
            f"step {step:3d}  train_l1={loss.item():.4f}  "
            f"valid_mae={valid['MAE']:.4f}  valid_acc2={valid['Acc2']:.3f}{marker}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_x, test_y)
    print()
    print("| split | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(format_metrics_row("synthetic MOSI test (tiny GMTM)", test))
    print()
    print(
        "This is not a published MOSI number. It only proves the Identity→GMTM→"
        "Identity stack and the metric helpers run on CPU."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
