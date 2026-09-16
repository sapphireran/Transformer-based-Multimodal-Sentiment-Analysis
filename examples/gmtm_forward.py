#!/usr/bin/env python3
"""Walk a synthetic clip through GatedMultiTransfomerModel and print shapes.

Mirrors the MOSEI-BERT GMTM call site:

    fusion = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=HParams)
    head = Identity()

so the fusion module itself emits the scalar sentiment score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import BERT_DIM, dummy_modalities
from models import AttentionPooling, GatedMultiTransfomerModel


class HParams:
    """Same knobs as ``model/train_GMTM_bert.py``."""

    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 64
    attn_mask = True
    output_dim = 1
    all_steps = False


def _project_one(model: GatedMultiTransfomerModel, stream: torch.Tensor, index: int) -> torch.Tensor:
    """Replicate the per-modality projection inside ``GatedMultiTransfomerModel.forward``."""
    xi = stream.permute(1, 0, 2)  # [T, B, F]
    time, batch, feat = xi.shape
    xi = xi.reshape(time * batch, feat)
    xi = model.modal_proj[index](xi)
    return xi.reshape(time, batch, model.embed_dim)


def describe_internal_shapes(model: GatedMultiTransfomerModel, streams: list[torch.Tensor]) -> None:
    """Print the tensors GMTM builds, without depending on private hooks."""
    proj = [_project_one(model, streams[i], i) for i in range(model.n_modalities)]
    print(f"  after modal_proj: {[tuple(p.shape) for p in proj]}  (T, B, embed)")

    pair_shapes = []
    hs = []
    for i in range(model.n_modalities):
        h_list = [model.trans[i][j](proj[i], proj[j], proj[j]) for j in range(model.n_modalities)]
        pair_shapes.append([tuple(h.shape) for h in h_list])
        weights = torch.softmax(model.modal_weights, dim=0)
        h_fused = torch.einsum("m,m...->...", weights, torch.stack(h_list, dim=0))
        gate = torch.sigmoid(model.gating_linears[i](h_fused))
        hs.append((gate * h_fused).permute(1, 0, 2))
    print(f"  cross-modal pair grid (q=i, k=j): {pair_shapes}")
    print(f"  learned modal_weights (softmax): {torch.softmax(model.modal_weights, dim=0).detach().tolist()}")
    print(f"  gated streams: {[tuple(h.shape) for h in hs]}  (B, T, embed)")

    concat = torch.cat(hs, dim=2)
    pooled = model.attn_pooling(concat)
    print(f"  concat over modalities: {tuple(concat.shape)}  (B, T, 3*embed)")
    print(f"  attention pool:         {tuple(pooled.shape)}  (B, 3*embed)")
    print(f"  classification head:    {tuple(model.classification_head(pooled).shape)}  (B, 1)")


def main() -> None:
    torch.manual_seed(0)
    visual, audio, text, labels = dummy_modalities(text_dim=BERT_DIM)
    print("Synthetic MOSEI-BERT clip")
    print(f"  visual {tuple(visual.shape)}  audio {tuple(audio.shape)}  text {tuple(text.shape)}")
    print(f"  labels {tuple(labels.shape)}  values={labels.squeeze(1).tolist()}")

    model = GatedMultiTransfomerModel(3, [35, 74, BERT_DIM], hyp_params=HParams)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGMTM params: {n_params:,}  embed_dim={HParams.embed_dim}  layers={HParams.layers}")

    print("\nInternal shapes")
    with torch.no_grad():
        describe_internal_shapes(model, [visual, audio, text])
        pred = model([visual, audio, text])
    print(f"\nforward() output: {tuple(pred.shape)}  scores={pred.squeeze(1).tolist()}")
    print(f"L1 vs dummy labels: {nn.L1Loss()(pred, labels).item():.4f}")
    print(
        "\nThis is an untrained network on random features — the loss is not a"
        " benchmark, only proof that the graph is wired the same way as"
        " train_GMTM_bert.py."
    )


if __name__ == "__main__":
    main()
