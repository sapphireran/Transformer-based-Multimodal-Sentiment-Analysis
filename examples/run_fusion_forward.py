#!/usr/bin/env python3
"""Forward-pass every fusion class on a synthetic BERT-shaped batch.

This is the shape checklist from docs/fusion-methods.md, executed against the
real modules in model/models.py. No pickle dumps, no GPU required.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn as nn

from device import describe_device, get_device
from paths import ensure_model_on_path, ensure_output_dir
from synthetic_affect import AffectShapes, batch_from_split, make_split_arrays

ensure_model_on_path()
from models import (  # noqa: E402
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
    TransformerSeq,
)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _as_cpu_list(tensors: List[torch.Tensor]) -> List[List[int]]:
    return [list(t.shape) for t in tensors]


def run(batch_size: int = 2, seq_len: int = 16, seed: int = 0) -> Dict:
    device = get_device()
    shapes = AffectShapes.for_embedding("bert", max_len=seq_len)
    split = make_split_arrays(batch_size, shapes=shapes, seed=seed)
    vision, audio, text, labels = batch_from_split(split, batch_size, device=device)
    sequences = [vision, audio, text]

    report: Dict[str, dict] = {
        "device": describe_device(device),
        "input_shapes": {
            "vision": list(vision.shape),
            "audio": list(audio.shape),
            "text": list(text.shape),
            "labels": list(labels.shape),
        },
        "modules": {},
    }

    def record(name: str, module: nn.Module, outputs: torch.Tensor, extra: dict | None = None) -> None:
        entry = {
            "params": _count_params(module),
            "output_shape": list(outputs.shape),
            "output_finite": bool(torch.isfinite(outputs).all().item()),
        }
        if extra:
            entry.update(extra)
        report["modules"][name] = entry
        print(
            f"{name:24s}  out={tuple(outputs.shape)!s:18s}  "
            f"params={entry['params']:<10d}  finite={entry['output_finite']}"
        )

    # --- vector fusions (late-style) -------------------------------------
    vis_vec = vision.mean(dim=1)
    aud_vec = audio.mean(dim=1)
    txt_vec = text.mean(dim=1)

    concat_late = ConcatLate().to(device)
    late_in = [
        torch.zeros(batch_size, 64, device=device),
        torch.zeros(batch_size, 256, device=device),
        torch.zeros(batch_size, 1024, device=device),
    ]
    # fill with pooled-then-projected noise so the cat is not all zeros
    late_in[0] = vis_vec[:, :64] if vis_vec.size(1) >= 64 else vis_vec.repeat(1, 2)[:, :64]
    # use a linear projection for the demo so widths match the BERT recipe
    proj = {
        0: nn.Linear(shapes.visual, 64).to(device),
        1: nn.Linear(shapes.audio, 256).to(device),
        2: nn.Linear(shapes.text, 1024).to(device),
    }
    late_in = [proj[i](v) for i, v in enumerate((vis_vec, aud_vec, txt_vec))]
    record("ConcatLate", concat_late, concat_late(late_in), extra={"expected": [batch_size, 1344]})

    tfn_proj = [
        nn.Linear(shapes.visual, 19).to(device),
        nn.Linear(shapes.audio, 39).to(device),
        nn.Linear(shapes.text, 159).to(device),
    ]
    tfn_in = [p(v) for p, v in zip(tfn_proj, (vis_vec, aud_vec, txt_vec))]
    tfn = TensorFusion().to(device)
    record("TensorFusion", tfn, tfn(tfn_in), extra={"expected": [batch_size, 128000]})

    lmf_proj = [
        nn.Linear(shapes.visual, 32).to(device),
        nn.Linear(shapes.audio, 64).to(device),
        nn.Linear(shapes.text, 256).to(device),
    ]
    lmf_in = [p(v) for p, v in zip(lmf_proj, (vis_vec, aud_vec, txt_vec))]
    lmf = LowRankTensorFusion([32, 64, 256], 256, 32).to(device)
    record("LowRankTensorFusion", lmf, lmf(lmf_in), extra={"expected": [batch_size, 256]})

    # --- sequence fusions ------------------------------------------------
    concat_early = ConcatEarly().to(device)
    early_out = concat_early(sequences)
    record("ConcatEarly", concat_early, early_out, extra={"expected": [batch_size, seq_len, 877]})

    early_tfm = EarlyFusionTransformer(n_features=877).to(device)
    record(
        "EarlyFusionTransformer",
        early_tfm,
        early_tfm(sequences),
        extra={"expected": [batch_size, 32], "note": "last timestep, embed_dim=32"},
    )

    seq_encoders = nn.ModuleList(
        [
            TransformerSeq(shapes.visual, 64),
            TransformerSeq(shapes.audio, 128),
            TransformerSeq(shapes.text, 256),
        ]
    ).to(device)
    encoded = [enc(x) for enc, x in zip(seq_encoders, sequences)]
    late_tfm = LateFusionTransformer(in_dim=64 + 128 + 256, embed_dim=32).to(device)
    record(
        "LateFusionTransformer",
        late_tfm,
        late_tfm(encoded),
        extra={
            "expected": [batch_size, 32],
            "encoder_shapes": _as_cpu_list(encoded),
            "note": "demo uses text width 256, not the 1024 used in train_main_bert.py",
        },
    )

    generic = TransformerFusion(d_model=64, nhead=4, num_layers=2).to(device)
    generic_in = [late_in[0], late_in[1][:, :64], nn.Linear(1024, 64).to(device)(late_in[2])]
    record("TransformerFusion", generic, generic(generic_in), extra={"expected": [batch_size, 64]})

    gmtm = GatedMultiTransfomerModel(3, [shapes.visual, shapes.audio, shapes.text]).to(device)
    record(
        "GMTM_default_hparams",
        gmtm,
        gmtm(sequences),
        extra={
            "expected": [batch_size, 1],
            "embed_dim": gmtm.embed_dim,
            "num_heads": gmtm.num_heads,
            "layers": gmtm.layers,
        },
    )

    mismatches = []
    for name, entry in report["modules"].items():
        if entry.get("expected") and entry["output_shape"] != entry["expected"]:
            mismatches.append((name, entry["output_shape"], entry["expected"]))
        if not entry["output_finite"]:
            mismatches.append((name, "non-finite", None))
    report["ok"] = not mismatches
    report["mismatches"] = [
        {"name": n, "got": g, "expected": e} for n, g, e in mismatches
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {describe_device()}")
    report = run(batch_size=args.batch_size, seq_len=args.seq_len, seed=args.seed)
    out_dir = ensure_output_dir()
    path = out_dir / "fusion_forward.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    if not report["ok"]:
        print("SHAPE MISMATCHES:", report["mismatches"])
        return 1
    print("all fusion output shapes match the documented contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
