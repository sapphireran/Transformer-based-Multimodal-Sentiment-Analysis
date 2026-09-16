#!/usr/bin/env python3
"""Forward-pass every fusion module on a tiny CPU batch.

Prints output shapes so a new machine can confirm ``models.py`` still
imports and the concat / tensor / transformer fusions agree with
``docs/architecture.md``.
"""

from __future__ import annotations

import json

from synthetic_data import (
    FeatureSpec,
    ensure_model_on_path,
    pick_device,
    random_padded_batch,
    seed_everything,
)

ensure_model_on_path()

import torch
from models import (
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
    TransformerSeq,
)


def _report(name: str, tensor: torch.Tensor) -> dict:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "mean": float(tensor.detach().float().mean().item()),
    }


def run() -> list[dict]:
    seed_everything(3)
    spec = FeatureSpec()
    device = pick_device()
    streams, _labels = random_padded_batch(batch_size=8, spec=spec, device=device)
    reports = []

    early = ConcatEarly().to(device)
    early_out = early(streams)
    reports.append(_report("ConcatEarly", early_out))
    assert early_out.shape == (8, spec.seq_len, spec.total)

    # Late concat expects already-pooled clip vectors.
    pooled = [s.mean(dim=1) for s in streams]
    late = ConcatLate().to(device)
    late_out = late(pooled)
    reports.append(_report("ConcatLate", late_out))
    assert late_out.shape == (8, spec.total)

    tf = TensorFusion().to(device)
    tf_out = tf(pooled)
    expected_tf = (spec.visual + 1) * (spec.audio + 1) * (spec.text + 1)
    reports.append(_report("TensorFusion", tf_out))
    assert tf_out.shape == (8, expected_tf)

    lrf = LowRankTensorFusion(spec.as_list, output_dim=12, rank=4).to(device)
    lrf_out = lrf(pooled)
    reports.append(_report("LowRankTensorFusion", lrf_out))
    assert lrf_out.shape == (8, 12)

    # TransformerFusion wants a shared width; project each clip vector.
    proj = [torch.nn.Linear(p.size(-1), 16).to(device) for p in pooled]
    shared = [layer(p) for layer, p in zip(proj, pooled)]
    fusion_tr = TransformerFusion(d_model=16, nhead=4, num_layers=1).to(device)
    tr_out = fusion_tr(shared)
    reports.append(_report("TransformerFusion", tr_out))
    assert tr_out.shape == (8, 16)

    eft = EarlyFusionTransformer(n_features=spec.total).to(device)
    eft_out = eft(streams)
    reports.append(_report("EarlyFusionTransformer", eft_out))
    assert eft_out.shape[-1] == EarlyFusionTransformer.embed_dim
    assert eft_out.shape[0] == 8

    encoded = [
        TransformerSeq(spec.visual, 16).to(device)(streams[0]),
        TransformerSeq(spec.audio, 16).to(device)(streams[1]),
        TransformerSeq(spec.text, 16).to(device)(streams[2]),
    ]
    lft = LateFusionTransformer(in_dim=48, embed_dim=16).to(device)
    lft_out = lft(encoded)
    reports.append(_report("LateFusionTransformer", lft_out))
    assert lft_out.shape == (8, 16)

    return reports


def main() -> None:
    reports = run()
    print(json.dumps({"device": str(pick_device()), "modules": reports}, indent=2))


if __name__ == "__main__":
    main()
