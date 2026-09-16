#!/usr/bin/env python3
"""Replay the GMTM modality-ablation API on synthetic clips.

``get_ablation_dataloader`` zeros the dropped streams instead of
removing them. This script does the same with ``zero_out_modalities``
and reports MAE for each keep-set. The ranking is only meaningful for
the synthetic label (a function of text) — on real MOSEI see
``docs/results.md``.

Run from the repository root:

    python examples/zero_modality_ablation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._bootstrap import ensure_output_dir

import torch

from metrics import evaluate_sentiment, format_score_table
from models import GatedMultiTransfomerModel
from synthetic import make_synthetic_batch, modality_dims, zero_out_modalities


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
    attn_mask = False
    output_dim = 1
    all_steps = False


COMBOS = [
    ("text", ("text",)),
    ("audio", ("audio",)),
    ("visual", ("visual",)),
    ("text+audio", ("text", "audio")),
    ("text+visual", ("text", "visual")),
    ("audio+visual", ("audio", "visual")),
    ("text+audio+visual", ("text", "audio", "visual")),
]


def main() -> None:
    torch.manual_seed(3)
    full = make_synthetic_batch(
        batch_size=16, seq_len=12, embedding="bert", seed=3, include_signal=True
    )
    model = GatedMultiTransfomerModel(3, list(modality_dims("bert")), hyp_params=TinyHParams)
    model.eval()

    rows = {}
    print("untrained GMTM, synthetic labels depend on text only\n")
    with torch.no_grad():
        for name, keep in COMBOS:
            batch = zero_out_modalities(full, keep)
            pred = model(batch.as_list())
            scores = evaluate_sentiment(batch.labels.numpy(), pred.numpy())
            rows[name] = scores
            print(
                f"{name:<20}  MAE={scores['MAE']:.4f}  "
                f"Corr={scores['Corr']:+.3f}  "
                f"mean logit={pred.mean().item():+.3f}"
            )

    print()
    print(format_score_table(rows))
    out = ensure_output_dir() / "synthetic_ablation.md"
    out.write_text(format_score_table(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*nested tensors.*")
    main()
