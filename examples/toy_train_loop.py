#!/usr/bin/env python3
"""Fit a tiny GMTM on synthetic labels so the train path is visible.

This is not MOSI/MOSEI. Features are Gaussian noise; labels are a noisy linear
function of the *text* mean so a 3-stream model can actually reduce L1 in a
few AdamW steps on CPU. Prints per-step train loss and a held-out MAE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.metrics import summarize_predictions
from examples.lib.paths import ensure_model_on_path
from examples.lib.synthetic import GLOVE_SPEC, make_batch

ensure_model_on_path()
from models import GatedMultiTransfomerModel  # noqa: E402


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
    attn_mask = True
    output_dim = 1
    all_steps = False


def _labeled_batch(n: int, seed: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """GloVe-width features; label ≈ tanh(mean(text)) scaled into [-3, 3]."""
    raw = make_batch(batch_size=n, spec=GLOVE_SPEC, seed=seed, label_mode="uniform")
    text_score = raw.text.mean(axis=(1, 2))
    labels = np.tanh(text_score) * 2.5
    labels = labels.astype(np.float32).reshape(-1, 1)
    xs = [
        torch.from_numpy(raw.vision),
        torch.from_numpy(raw.audio),
        torch.from_numpy(raw.text),
    ]
    return xs, torch.from_numpy(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = GatedMultiTransfomerModel(3, GLOVE_SPEC.dims, hyp_params=TinyHParams)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.L1Loss()

    train_x, train_y = _labeled_batch(args.batch_size, seed=args.seed)
    valid_x, valid_y = _labeled_batch(args.batch_size, seed=args.seed + 99)

    history: List[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        opt.step()
        history.append(float(loss.item()))
        print(f"step {step:02d}/{args.steps}  train_L1={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        valid_pred = model(valid_x)
    metrics = summarize_predictions(valid_y.numpy(), valid_pred.numpy())
    print(
        "held-out  MAE={MAE:.4f}  Acc2={Acc2:.4f}  Corr={Corr:.4f}".format(**metrics)
    )
    print(f"loss path: {', '.join(f'{v:.4f}' for v in history)}")

    if history[-1] >= history[0]:
        print(
            "note: last train L1 was not below the first step "
            "(possible on a tiny random run; rerun or raise --steps)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
