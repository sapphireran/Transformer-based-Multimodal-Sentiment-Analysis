#!/usr/bin/env python3
"""Walk a synthetic clip through GMTM and print every intermediate rank.

Uses the same ``(B, T, F)`` layout as the padded MOSEI / MOSI loaders.
"""

from __future__ import annotations

import json

from synthetic_data import (
    FeatureSpec,
    ensure_model_on_path,
    pick_device,
    random_padded_batch,
    seed_everything,
    zero_ablate,
)

ensure_model_on_path()

import torch
from models import GatedMultiTransfomerModel


class ToyHParams:
    num_heads = 4
    layers = 2
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


def build_model(spec: FeatureSpec, device: torch.device) -> GatedMultiTransfomerModel:
    model = GatedMultiTransfomerModel(3, spec.as_list, hyp_params=ToyHParams)
    return model.to(device)


def describe_forward(model: GatedMultiTransfomerModel, streams):
    """Replay the GMTM body with shape notes (mirrors ``models.py``)."""
    notes = []
    proj_x = []
    for i, stream in enumerate(streams):
        notes.append({"stage": f"input[{i}]", "shape": list(stream.shape)})
        xi = stream.permute(1, 0, 2)
        t, b, f = xi.size()
        xi = model.modal_proj[i](xi.reshape(t * b, f)).reshape(t, b, model.embed_dim)
        notes.append({"stage": f"modal_proj[{i}]", "shape": list(xi.shape)})
        proj_x.append(xi)

    hs = []
    for i in range(model.n_modalities):
        h_list = []
        for j in range(model.n_modalities):
            h_ij = model.trans[i][j](proj_x[i], proj_x[j], proj_x[j])
            notes.append({"stage": f"cross[{i}->{j}]", "shape": list(h_ij.shape)})
            h_list.append(h_ij)
        weights = torch.softmax(model.modal_weights, dim=0)
        h_fused = torch.einsum("m,m...->...", weights, torch.stack(h_list, dim=0))
        gate = torch.sigmoid(model.gating_linears[i](h_fused))
        h_fused = gate * h_fused
        hs.append(h_fused.permute(1, 0, 2))
        notes.append({"stage": f"gated[{i}]", "shape": list(hs[-1].shape)})

    cat = torch.cat(hs, dim=2)
    notes.append({"stage": "concat", "shape": list(cat.shape)})
    pooled = model.attn_pooling(cat)
    notes.append({"stage": "attn_pool", "shape": list(pooled.shape)})
    out = model.classification_head(pooled)
    notes.append({"stage": "head", "shape": list(out.shape)})
    return out, notes


def run() -> dict:
    seed_everything(5)
    spec = FeatureSpec()
    device = pick_device()
    streams, labels = random_padded_batch(batch_size=6, spec=spec, device=device)
    model = build_model(spec, device)
    model.eval()

    with torch.no_grad():
        out, notes = describe_forward(model, streams)
        direct = model(streams)
        ablated = model(zero_ablate(streams, keep=("text",)))

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "device": str(device),
        "n_params": n_params,
        "label_shape": list(labels.shape),
        "forward_matches_replay": bool(torch.allclose(out, direct, atol=1e-5)),
        "full_out_mean": float(direct.mean().item()),
        "text_only_out_mean": float(ablated.mean().item()),
        "stages": notes,
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
