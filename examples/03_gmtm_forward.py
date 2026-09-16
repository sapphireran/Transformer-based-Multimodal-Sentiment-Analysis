#!/usr/bin/env python3
"""CPU forward pass through ``GatedMultiTransfomerModel``.

Uses a shrunken HParams block (2 layers, embed 16) so the 3×3 cross-modal
grid finishes in a few seconds without a GPU. Widths stay MOSI/MOSEI-real
(35 / 74 / 768) so the projection layers match the training scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import make_aligned_batch
from models import GatedMultiTransfomerModel, Identity


class TinyHParams:
    num_heads = 2
    layers = 2
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


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    torch.manual_seed(1)
    vision, audio, text, labels = make_aligned_batch(batch_size=2, seq_len=8)
    n_features = [vision.size(-1), audio.size(-1), text.size(-1)]

    gmtm = GatedMultiTransfomerModel(n_modalities=3, n_features=n_features, hyp_params=TinyHParams)
    gmtm.eval()

    # Same wrapper the training scripts use: Identity encoders + Identity head.
    encoders = [Identity(), Identity(), Identity()]
    with torch.no_grad():
        reps = [enc(x) for enc, x in zip(encoders, (vision, audio, text))]
        pred = gmtm(reps)

    print("GMTM input widths", n_features)
    print("prediction", tuple(pred.shape), pred.flatten().tolist())
    print("labels     ", labels.flatten().tolist())
    print(f"parameters {count_params(gmtm):,}")
    print(f"modal softmax weights {torch.softmax(gmtm.modal_weights, dim=0).tolist()}")
    print(f"alpha (unused residual mix) {gmtm.alpha.item():.4f}")
    print(f"cross-modal grid  {len(gmtm.trans)} x {len(gmtm.trans[0])} encoders")
    print(f"memory stacks built but unused: {len(gmtm.trans_mems)}")
    assert pred.shape == (2, 1)
    assert torch.isfinite(pred).all()
    print()
    print("Forward pass ok. Full HParams (embed 64, 4 layers, 4 heads) live in train_GMTM_*.py.")


if __name__ == "__main__":
    main()
