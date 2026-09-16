"""Gated multi-transformer forward pass and a zero-mask ablation.

Mirrors ``get_ablation_dataloader``: unused modalities become zeros, the
graph always has three towers. That is why text-only and trimodal GMTM
share a parameter count in the real CSVs.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

import torch

from common import AUDIO_DIM, TEXT_GLOVE_DIM, VISUAL_DIM, count_params, device, finite
from synthetic_multimodal import random_batch, zero_mask

import models as M


class TinyHParams:
    """Smaller than the training HParams so a CPU pass finishes quickly."""

    num_heads = 2
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


COMBINATIONS: List[Sequence[str]] = [
    ("text",),
    ("audio",),
    ("visual",),
    ("text", "audio"),
    ("text", "visual"),
    ("audio", "visual"),
    ("text", "audio", "visual"),
]


def build_gmtm(text_dim: int = TEXT_GLOVE_DIM, dev: Optional[torch.device] = None):
    dev = dev or device()
    model = M.GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[VISUAL_DIM, AUDIO_DIM, text_dim],
        hyp_params=TinyHParams,
    ).to(dev)
    return model


def run(verbose: bool = True) -> Dict[str, dict]:
    dev = device()
    torch.manual_seed(1)
    text_dim = 32  # shrink BERT/GloVe just for the demo
    model = build_gmtm(text_dim=text_dim, dev=dev)
    model.eval()
    vision, audio, text = random_batch(batch_size=4, seq_len=10, text_dim=text_dim, device=dev)

    results = {}
    params = count_params(model)
    if verbose:
        print(f"device={dev}  GMTM params={params:,}  (constant across masks)")

    with torch.no_grad():
        for keep in COMBINATIONS:
            v, a, t = zero_mask(vision, audio, text, keep)
            out = model([v, a, t])
            if out.dim() == 1:
                out = out.view(-1, 1)
            ok = finite(out) and out.shape == (vision.shape[0], 1)
            key = "+".join(keep)
            results[key] = {
                "out_shape": tuple(out.shape),
                "mean": float(out.mean().cpu()),
                "std": float(out.std(unbiased=False).cpu()),
                "ok": ok,
                "params": params,
            }
            if verbose:
                status = "ok" if ok else "FAIL"
                print(
                    f"[{status}] {key:20s}  mean={results[key]['mean']:+.4f}  "
                    f"std={results[key]['std']:.4f}"
                )
    return results


def main() -> int:
    rows = run(verbose=True)
    failed = [k for k, v in rows.items() if not v["ok"]]
    param_set = {v["params"] for v in rows.values()}
    if len(param_set) != 1:
        print("parameter count changed across masks — unexpected", file=sys.stderr)
        return 1
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    print("GMTM ablation demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
