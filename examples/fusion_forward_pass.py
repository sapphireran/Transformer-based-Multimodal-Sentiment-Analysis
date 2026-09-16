#!/usr/bin/env python3
"""Run every bake-off fusion module on a synthetic MOSI/MOSEI batch.

Uses the real classes in `model/models.py`. Default device is CPU. This is a
shape / smoke test, not a trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.paths import ensure_model_on_path
from examples.lib.synthetic import BERT_SPEC, GLOVE_SPEC, FeatureSpec, make_batch

ensure_model_on_path()
from models import (  # noqa: E402
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
)


def _tensors(batch, device: torch.device) -> List[torch.Tensor]:
    return [
        torch.from_numpy(batch.vision).to(device),
        torch.from_numpy(batch.audio).to(device),
        torch.from_numpy(batch.text).to(device),
    ]


def _mean_pool(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.mean(dim=1) for x in xs]


def run_fusions(spec: FeatureSpec, batch_size: int, seed: int, device: torch.device) -> List[Tuple[str, tuple]]:
    batch = make_batch(batch_size=batch_size, spec=spec, seed=seed)
    xs = _tensors(batch, device)
    pooled = _mean_pool(xs)
    shapes: List[Tuple[str, tuple]] = []

    cases: List[Tuple[str, Callable[[], torch.Tensor]]] = [
        ("ConcatEarly [B,T,F]", lambda: ConcatEarly()(xs)),
        ("ConcatLate (time-flattened)", lambda: ConcatLate()(xs)),
        ("ConcatLate (already pooled)", lambda: ConcatLate()(pooled)),
        ("TensorFusion (pooled + ones)", lambda: TensorFusion()(pooled)),
        (
            "LowRankTensorFusion rank=8",
            lambda: LowRankTensorFusion(
                [spec.visual, spec.audio, spec.text], output_dim=32, rank=8, flatten=False
            ).to(device)(pooled),
        ),
        (
            "TransformerFusion on pooled vectors",
            lambda: TransformerFusion(d_model=spec.visual, nhead=5, num_layers=1).to(device)(
                # project each pooled stream to visual_dim so they share d_model
                [
                    torch.nn.functional.adaptive_avg_pool1d(p.unsqueeze(1), spec.visual).squeeze(1)
                    if p.size(-1) != spec.visual
                    else p
                    for p in pooled
                ]
            ),
        ),
        (
            "EarlyFusionTransformer last-step",
            lambda: EarlyFusionTransformer(n_features=spec.total_early).to(device)(xs),
        ),
        (
            "LateFusionTransformer last-step",
            lambda: LateFusionTransformer(in_dim=spec.total_early, embed_dim=32).to(device)(xs),
        ),
    ]

    for name, fn in cases:
        module_out = fn()
        shapes.append((name, tuple(module_out.shape)))
        print(f"{name:40s} {tuple(module_out.shape)}")
    return shapes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", choices=("bert", "glove"), default="bert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = BERT_SPEC if args.text == "bert" else GLOVE_SPEC
    device = torch.device("cpu")
    print(f"spec={spec.name} dims={spec.dims} batch={args.batch_size} device={device}")
    print(f"synthetic shapes: {make_batch(args.batch_size, spec, args.seed).shapes()}")
    print()
    with torch.no_grad():
        run_fusions(spec, args.batch_size, args.seed, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
