#!/usr/bin/env python3
"""Tiny CPU training loop on synthetic, text-correlated sentiment.

Mirrors ``MultiFramework`` + ConcatLate + L1 + AdamW from the real
scripts, with small LSTMs and 8 epochs. The label is a noisy function of
the mean text channel (see ``common.make_aligned_batch(correlated=True)``),
so MAE should fall if the late-fusion head is actually learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BERT_DIM, make_aligned_batch
from models import ConcatLate, LSTM, MLP
from train_and_test import MultiFramework


def make_model() -> MultiFramework:
    # Identity vision/audio (the signal is in text) + a small text LSTM.
    encoders = [
        LSTM(35, 16, dropout=False, has_padding=False),
        LSTM(74, 16, dropout=False, has_padding=False),
        LSTM(BERT_DIM, 32, dropout=False, has_padding=False),
    ]
    fusion = ConcatLate()
    head = MLP(16 + 16 + 32, 32, 1)
    return MultiFramework(encoders, fusion, head, has_padding=False)


def batch_iter(n_batches: int, batch_size: int, seq_len: int, generator: torch.Generator):
    for _ in range(n_batches):
        yield make_aligned_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            text_dim=BERT_DIM,
            correlated=True,
            generator=generator,
        )


def run_epoch(model: MultiFramework, optimizer, n_batches: int, batch_size: int, seq_len: int, generator, train: bool) -> float:
    criterion = nn.L1Loss()
    total = 0.0
    count = 0
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for vision, audio, text, labels in batch_iter(n_batches, batch_size, seq_len, generator):
            pred = model([vision, audio, text])
            loss = criterion(pred, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 8.0)
                optimizer.step()
            total += loss.item() * labels.size(0)
            count += labels.size(0)
    return total / count


def main() -> None:
    torch.manual_seed(7)
    train_gen = torch.Generator().manual_seed(11)
    valid_gen = torch.Generator().manual_seed(23)

    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    print("epoch  train_MAE  valid_MAE")

    history = []
    for epoch in range(1, 9):
        train_mae = run_epoch(model, opt, n_batches=12, batch_size=8, seq_len=8, generator=train_gen, train=True)
        valid_mae = run_epoch(model, opt, n_batches=4, batch_size=8, seq_len=8, generator=valid_gen, train=False)
        history.append((epoch, train_mae, valid_mae))
        print(f"{epoch:5d}  {train_mae:9.4f}  {valid_mae:9.4f}")

    first_valid = history[0][2]
    last_valid = history[-1][2]
    print()
    print(f"valid MAE  epoch1={first_valid:.4f}  epoch8={last_valid:.4f}  "
          f"delta={first_valid - last_valid:+.4f}")
    if last_valid < first_valid:
        print("Validation MAE fell — ConcatLate is fitting the synthetic text signal.")
    else:
        print("Validation MAE did not fall; rerun or bump epochs. The loop itself still ran.")

    # One held-out batch for a sanity print of pred vs label.
    vision, audio, text, labels = make_aligned_batch(
        batch_size=4, seq_len=8, correlated=True, generator=torch.Generator().manual_seed(99)
    )
    model.eval()
    with torch.no_grad():
        pred = model([vision, audio, text])
    print("sample pred ", [round(x, 3) for x in pred.flatten().tolist()])
    print("sample label", [round(x, 3) for x in labels.flatten().tolist()])


if __name__ == "__main__":
    main()
