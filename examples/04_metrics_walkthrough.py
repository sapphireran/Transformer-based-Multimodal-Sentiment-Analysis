#!/usr/bin/env python3
"""Walk through MAE / Acc-7 / Acc-5 / Acc-2 / F1 on toy predictions.

    python examples/04_metrics_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from metrics import (  # noqa: E402
    evaluate_sentiment,
    format_score_table,
    split_uniform_7,
    summarize_bin_edges,
)
from msa_lab.synthetic import make_sentiment_batch  # noqa: E402


def noisy_copy(labels: torch.Tensor, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(labels.shape, generator=generator) * scale
    return torch.clamp(labels + noise, -3.0, 3.0)


def main() -> None:
    batch = make_sentiment_batch(batch_size=32, seq_len=8, preset="toy", seed=3)
    y = batch.labels
    candidates = [
        ("identity (oracle)", y),
        ("light noise", noisy_copy(y, 0.25, seed=4)),
        ("heavy noise", noisy_copy(y, 1.40, seed=5)),
        ("sign flip", -y),
    ]

    print("Uniform Acc-7 bins over [-3, 3]:")
    for label, left, right in summarize_bin_edges(7):
        print(f"  bin {label}: [{left: .3f}, {right: .3f})")
    print()

    named = []
    for name, pred in candidates:
        scores = evaluate_sentiment(y, pred)
        named.append((name, scores))

    print(format_score_table(named))
    print()
    print("First eight oracle labels and their Acc-7 bins:")
    labels = y.squeeze(-1)[:8]
    bins = split_uniform_7(labels)
    for score, bucket in zip(labels.tolist(), bins.tolist()):
        print(f"  {score:6.3f} -> bin {int(bucket)}")


if __name__ == "__main__":
    main()
