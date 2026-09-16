#!/usr/bin/env python3
"""Zero-out modalities the way the ablation dataloaders do.

``get_ablation_dataloader`` keeps a three-slot batch and writes zeros
into dropped streams so GMTM can keep a fixed ``n_modalities=3``.

    python examples/06_ablation_zero_modalities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from msa_lab.fusion_zoo import build_fusion_zoo  # noqa: E402
from msa_lab.synthetic import make_sentiment_batch, zero_modalities  # noqa: E402
from metrics import evaluate_sentiment, format_score_table  # noqa: E402


def energy(tensor: torch.Tensor) -> float:
    return float(tensor.pow(2).mean().cpu())


def main() -> None:
    full = make_sentiment_batch(batch_size=12, seq_len=10, preset="toy", seed=13, noise=0.05)
    zoo = {spec.name: spec for spec in build_fusion_zoo(full.dims())}
    gated = zoo["GatedMultiTransformer"]
    gated.module.eval()

    combinations = [
        ("text+audio+visual", []),
        ("text+audio", ["visual"]),
        ("text+visual", ["audio"]),
        ("audio+visual", ["text"]),
        ("text", ["audio", "visual"]),
        ("audio", ["text", "visual"]),
        ("visual", ["text", "audio"]),
    ]

    named = []
    print(f"{'combo':<22}  vis-E    aud-E    txt-E    pred-std")
    print("-" * 62)
    with torch.no_grad():
        for name, dropped in combinations:
            batch = zero_modalities(full, dropped)
            pred = gated.prepare(batch)
            print(
                f"{name:<22}  {energy(batch.visual):7.4f}  {energy(batch.audio):7.4f}  "
                f"{energy(batch.text):7.4f}  {float(pred.std()):8.4f}"
            )
            named.append((name, evaluate_sentiment(full.labels, pred)))

    print()
    print("Untrained GMTM scores (expect near-chance; this is a wiring demo):")
    print(format_score_table(named))


if __name__ == "__main__":
    main()
