#!/usr/bin/env python3
"""Overfit GatedMultiTransfomerModel on one synthetic batch (CPU).

This is a sanity check that the gated cross-modal stack is trainable, not
a reproduction of ``model/results/ablation_results.csv``.

    python examples/03_gated_transformer_toy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from msa_lab.toy_train import train_gated_transformer  # noqa: E402


def main() -> None:
    result = train_gated_transformer(steps=16, batch_size=16, seq_len=12, seed=11)
    print("Gated Multi-Transformer toy overfit")
    print(f"  device       {result.device}")
    print(f"  steps        {result.steps}")
    print(f"  initial MAE  {result.initial_mae:.4f}")
    print(f"  final MAE    {result.final_mae:.4f}")
    print(f"  improved     {result.improved}")
    preview = ", ".join(f"{value:.3f}" for value in result.history[:8])
    print(f"  loss start   {preview}")
    if not result.improved:
        raise SystemExit("toy train failed to reduce MAE; check GMTM forward path")
    drop = result.initial_mae - result.final_mae
    print(f"  MAE drop     {drop:.4f}")


if __name__ == "__main__":
    main()
