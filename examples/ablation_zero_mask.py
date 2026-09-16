#!/usr/bin/env python3
"""Show the ablation protocol: unused streams are zeros, GMTM stays 3-wide.

``get_ablation_dataloader`` does not shrink ``n_modalities``. A text-only run
is a full GMTM whose audio and visual inputs are ``torch.zeros(50, F)``.
This script rebuilds that mask on a synthetic batch and compares predictions
against the same untrained weights on the full trio — so any gap is caused
only by zeroing streams, not by a different module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import AUDIO_DIM, BERT_DIM, VISUAL_DIM, dummy_modalities
from models import GatedMultiTransfomerModel


class TinyHParams:
    num_heads = 4
    layers = 2
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 32
    attn_mask = True
    output_dim = 1
    all_steps = False


COMBOS = [
    ["text"],
    ["audio"],
    ["visual"],
    ["text", "audio"],
    ["text", "visual"],
    ["audio", "visual"],
    ["text", "audio", "visual"],
]

INDEX = {"visual": 0, "audio": 1, "text": 2}


def mask_streams(streams: list[torch.Tensor], keep: list[str]) -> list[torch.Tensor]:
    kept = set(keep)
    out = []
    names = ["visual", "audio", "text"]
    for name, tensor in zip(names, streams):
        out.append(tensor if name in kept else torch.zeros_like(tensor))
    return out


def main() -> None:
    torch.manual_seed(0)
    visual, audio, text, labels = dummy_modalities(batch=4, seq_len=12, text_dim=BERT_DIM, seed=7)
    full = [visual, audio, text]
    print("Official ablation widths (max_pad path):")
    print(f"  visual {VISUAL_DIM}  audio {AUDIO_DIM}  text {BERT_DIM}  (always three tensors)")
    print(f"  full-batch norms: vis={visual.norm():.2f} aud={audio.norm():.2f} txt={text.norm():.2f}")

    model = GatedMultiTransfomerModel(3, [VISUAL_DIM, AUDIO_DIM, BERT_DIM], hyp_params=TinyHParams)
    model.eval()
    criterion = nn.L1Loss()

    print(f"\n{'keep':<28} {'zeroed':<24} {'||pred||':>10} {'L1 vs y':>10} {'Δ vs full':>10}")
    with torch.no_grad():
        full_pred = model(full)
        full_l1 = criterion(full_pred, labels).item()
        rows = []
        for keep in COMBOS:
            masked = mask_streams(full, keep)
            zeroed = [name for name in ("visual", "audio", "text") if name not in keep]
            pred = model(masked)
            l1 = criterion(pred, labels).item()
            delta = torch.mean(torch.abs(pred - full_pred)).item()
            rows.append((keep, zeroed, pred.norm().item(), l1, delta, pred))
            print(
                f"{'+'.join(keep):<28} {','.join(zeroed) or '—':<24} "
                f"{pred.norm().item():10.4f} {l1:10.4f} {delta:10.4f}"
            )

    print(
        f"\nFull-trio L1 against dummy labels: {full_l1:.4f}"
        " (untrained; use the column Δ vs full to see the mask effect)."
    )
    print(
        "On the recorded MOSEI BERT table, audio-only / visual-only sit near"
        " MAE 0.82 while text-only is 0.5687 — the same zero-mask protocol,"
        " with trained weights."
    )
    # Sanity: masking nothing must match the full forward exactly.
    assert torch.allclose(rows[-1][5], full_pred)


if __name__ == "__main__":
    main()
