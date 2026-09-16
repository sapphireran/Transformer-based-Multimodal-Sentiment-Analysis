#!/usr/bin/env python3
"""Forward a randomly-initialized GMTM on a synthetic 3-stream batch.

`--ablate text` (or audio / visual / pairs) zeros the other streams the same
way `get_ablation_dataloader` does. Predictions are meaningless (no weights);
the point is the tensor path and the ablation mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.paths import ensure_model_on_path
from examples.lib.synthetic import (
    BERT_SPEC,
    GLOVE_SPEC,
    ablate_modalities,
    iter_ablation_sets,
    make_batch,
)

ensure_model_on_path()
from models import GatedMultiTransfomerModel  # noqa: E402


class DemoHParams:
    """Smaller than the recorded 4×64 stack so a CPU forward is cheap."""

    num_heads = 4
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


def _parse_keep(raw: str | None) -> List[str]:
    if raw is None:
        return ["text", "audio", "visual"]
    parts = [p.strip() for p in raw.replace("+", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --ablate list")
    return parts


def build_model(text: str, device: torch.device) -> GatedMultiTransfomerModel:
    spec = BERT_SPEC if text == "bert" else GLOVE_SPEC
    model = GatedMultiTransfomerModel(3, spec.dims, hyp_params=DemoHParams)
    return model.to(device).eval()


def forward_once(
    text: str,
    keep: List[str],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    spec = BERT_SPEC if text == "bert" else GLOVE_SPEC
    batch = ablate_modalities(make_batch(batch_size, spec, seed), keep)
    xs = [
        torch.from_numpy(batch.vision).to(device),
        torch.from_numpy(batch.audio).to(device),
        torch.from_numpy(batch.text).to(device),
    ]
    model = build_model(text, device)
    with torch.no_grad():
        return model(xs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", choices=("bert", "glove"), default="bert")
    parser.add_argument(
        "--ablate",
        default=None,
        help="modalities to KEEP, e.g. 'text' or 'text,audio' (default: all three)",
    )
    parser.add_argument("--all-combos", action="store_true", help="run every 7-way ablation")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    device = torch.device("cpu")
    combos = list(iter_ablation_sets()) if args.all_combos else [_parse_keep(args.ablate)]

    for keep in combos:
        pred = forward_once(args.text, keep, args.batch_size, args.seed, device)
        print(
            f"keep={keep!s:30s} out={tuple(pred.shape)} "
            f"preds={pred.squeeze(-1).tolist()}"
        )
        if pred.ndim != 2 or pred.shape != (args.batch_size, 1):
            raise SystemExit(f"unexpected GMTM shape {tuple(pred.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
