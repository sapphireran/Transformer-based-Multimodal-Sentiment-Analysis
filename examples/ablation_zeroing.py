"""Reproduce the loader's unimodal zero-fill and run GMTM on each subset."""

from __future__ import annotations

from typing import Sequence

import torch

from examples.common import TinyGMTMParams, cpu_device, finite
from examples.forward_gmtm import build_gmtm
from examples.synthetic_data import make_batch, make_unimodal_batch

SUBSETS: list[list[str]] = [
    ["text"],
    ["audio"],
    ["visual"],
    ["text", "audio"],
    ["text", "visual"],
    ["audio", "visual"],
    ["text", "audio", "visual"],
]


def score_subsets(
    keep_lists: Sequence[Sequence[str]] | None = None,
    seed: int = 3,
) -> list[dict]:
    keep_lists = list(keep_lists or SUBSETS)
    device = cpu_device()
    model = build_gmtm(hyp_params=TinyGMTMParams).to(device)
    model.eval()
    full = make_batch(batch_size=4, seq_len=10, seed=seed)
    rows = []
    with torch.no_grad():
        for keep in keep_lists:
            batch = make_unimodal_batch(keep, full)
            out = model([t.to(device) for t in batch.as_list()])
            if not finite(out):
                raise RuntimeError(f"non-finite output for {keep}")
            rows.append(
                {
                    "modalities": "+".join(keep),
                    "n_kept": len(keep),
                    "out_std": float(out.std()),
                    "out_mean": float(out.mean()),
                    "nonzero_vision": bool(batch.vision.abs().sum() > 0),
                    "nonzero_audio": bool(batch.audio.abs().sum() > 0),
                    "nonzero_text": bool(batch.text.abs().sum() > 0),
                }
            )
    return rows


def main() -> int:
    print("GMTM still sees 3 tensors; dropped modalities are zeros")
    print("(random init — this is an API demo, not a trained ablation)")
    rows = score_subsets()
    for row in rows:
        flags = (
            f"V={'on' if row['nonzero_vision'] else '0'} "
            f"A={'on' if row['nonzero_audio'] else '0'} "
            f"T={'on' if row['nonzero_text'] else '0'}"
        )
        print(
            f"  {row['modalities']:<22} {flags}  "
            f"mean={row['out_mean']:+.4f}  std={row['out_std']:.4f}"
        )
    # Sanity: the all-zero audio+visual row must actually zero those slots.
    three = next(r for r in rows if r["modalities"] == "text")
    if three["nonzero_audio"] or three["nonzero_vision"] or not three["nonzero_text"]:
        raise RuntimeError("text-only subset did not zero audio/vision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
