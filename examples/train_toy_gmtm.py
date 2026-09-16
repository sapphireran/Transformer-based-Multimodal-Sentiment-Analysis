#!/usr/bin/env python3
"""Tiny CPU training loop for GatedMultiTransfomerModel.

Uses synthetic clips whose labels are a noisy function of the text
stream (see ``model/synthetic.py``). This is a walk-through of the
GMTM API, not a MOSI/MOSEI result.

Run from the repository root:

    python examples/train_toy_gmtm.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._bootstrap import ensure_output_dir

import torch
import torch.nn as nn

from metrics import evaluate_sentiment, format_score_table
from models import GatedMultiTransfomerModel
from synthetic import make_synthetic_split, modality_dims


class TinyHParams:
    """Smaller than the paper HParams so a CPU can finish in seconds."""

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


def _run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    preds = []
    truths = []
    for visual, audio, text, labels in loader:
        out = model([visual, audio, text])
        loss = nn.functional.l1_loss(out, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        preds.append(out.detach())
        truths.append(labels.detach())
    stacked_pred = torch.cat(preds, dim=0)
    stacked_true = torch.cat(truths, dim=0)
    scores = evaluate_sentiment(stacked_true.numpy(), stacked_pred.numpy())
    scores["L1"] = total_loss / max(n, 1)
    return scores


def main() -> None:
    torch.manual_seed(11)
    loaders = make_synthetic_split(
        n_train=48, n_valid=16, n_test=16, batch_size=8, seq_len=12, embedding="bert", seed=11
    )
    model = GatedMultiTransfomerModel(3, list(modality_dims("bert")), hyp_params=TinyHParams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print("toy GMTM  embed_dim=16  layers=1  seq_len=12  device=cpu")
    history = []
    for epoch in range(1, 6):
        train_scores = _run_epoch(model, loaders["train"], optimizer)
        valid_scores = _run_epoch(model, loaders["valid"], optimizer=None)
        history.append((epoch, train_scores["L1"], valid_scores["L1"]))
        print(
            f"epoch {epoch}  train L1={train_scores['L1']:.4f}  "
            f"valid L1={valid_scores['L1']:.4f}  "
            f"valid MAE={valid_scores['MAE']:.4f}  "
            f"valid Corr={valid_scores['Corr']:.3f}"
        )

    test_scores = _run_epoch(model, loaders["test"], optimizer=None)
    print("\ntest metrics (synthetic, not MOSI/MOSEI):")
    print(format_score_table({"toy-GMTM": test_scores}))

    out_dir = ensure_output_dir()
    hist_path = out_dir / "toy_gmtm_history.csv"
    lines = ["epoch,train_l1,valid_l1"]
    lines.extend(f"{e},{tr:.6f},{va:.6f}" for e, tr, va in history)
    hist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ckpt = out_dir / "toy_gmtm.pt"
    torch.save({"state_dict": model.state_dict(), "hparams": "TinyHParams"}, ckpt)
    print(f"\nwrote {hist_path}")
    print(f"wrote {ckpt}")

    if history[-1][1] < history[0][1]:
        print("train L1 fell across the 5 toy epochs (expected on this synthetic signal).")
    else:
        print("train L1 did not fall in 5 epochs; rerun or add steps — still a valid smoke test.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*nested tensors.*")
    main()
