#!/usr/bin/env python3
"""Run every fusion module used in the trainers on one synthetic batch.

This does not load checkpoints. It only checks that the classes in
``model/models.py`` accept MOSI-shaped tensors and return a fused vector.

    python examples/02_fusion_forward_pass.py
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

from common import (
    TINY_AUDIO_DIM,
    TINY_SEQ_LEN,
    TINY_TEXT_DIM,
    TINY_VISION_DIM,
    TinyHParams,
    count_parameters,
    device,
    make_synthetic_mosi,
)

from models import (  # noqa: E402  (path setup in common)
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    LateFusionTransformer,
    LowRankTensorFusion,
    TensorFusion,
    TransformerSeq,
)


def _batch(dev: torch.device):
    split = make_synthetic_mosi(n_train=4, n_valid=1, n_test=1, seed=1)["train"]
    vision = torch.from_numpy(split["vision"]).float().to(dev)
    audio = torch.from_numpy(split["audio"]).float().to(dev)
    text = torch.from_numpy(split["text"]).float().to(dev)
    return vision, audio, text


def run_concat_early(vision, audio, text):
    fused = ConcatEarly()([vision, audio, text])
    return fused, {"fused": tuple(fused.shape)}


def run_concat_late(vision, audio, text):
    # Late concat in the trainers sees *pooled* encoder states. Flatten time here.
    mods = [m.mean(dim=1) for m in (vision, audio, text)]
    fused = ConcatLate()(mods)
    return fused, {"fused": tuple(fused.shape)}


def run_tensor_fusion(vision, audio, text):
    mods = [m.mean(dim=1)[:, :8] for m in (vision, audio, text)]
    fused = TensorFusion()(mods)
    return fused, {"fused": tuple(fused.shape)}


def run_low_rank(vision, audio, text):
    dims = [8, 8, 8]
    mods = [m.mean(dim=1)[:, : d] for m, d in zip((vision, audio, text), dims)]
    fused = LowRankTensorFusion(dims, output_dim=16, rank=4)(mods)
    return fused, {"fused": tuple(fused.shape)}


def run_transformer_early(vision, audio, text):
    n_features = vision.size(-1) + audio.size(-1) + text.size(-1)
    model = EarlyFusionTransformer(n_features=n_features)
    fused = model([vision, audio, text])
    return fused, {"fused": tuple(fused.shape)}


def run_transformer_late(vision, audio, text):
    encoders = [
        TransformerSeq(TINY_VISION_DIM, 16),
        TransformerSeq(TINY_AUDIO_DIM, 16),
        TransformerSeq(TINY_TEXT_DIM, 16),
    ]
    encoded = [enc(x) for enc, x in zip(encoders, (vision, audio, text))]
    in_dim = 16 * 3
    fused = LateFusionTransformer(in_dim=in_dim, embed_dim=16)(encoded)
    return fused, {"fused": tuple(fused.shape)}


def run_gmtm(vision, audio, text):
    model = GatedMultiTransfomerModel(
        3,
        [TINY_VISION_DIM, TINY_AUDIO_DIM, TINY_TEXT_DIM],
        hyp_params=TinyHParams,
    )
    fused = model([vision, audio, text])
    return fused, {"fused": tuple(fused.shape), "params": count_parameters(model)}


RUNNERS: Dict[str, Callable] = {
    "ConcatEarly": run_concat_early,
    "ConcatLate": run_concat_late,
    "TensorFusion": run_tensor_fusion,
    "LowRankTensorFusion": run_low_rank,
    "TransformerEarly": run_transformer_early,
    "TransformerLate": run_transformer_late,
    "GatedMultiTransfomer": run_gmtm,
}


def main() -> int:
    dev = device()
    vision, audio, text = _batch(dev)
    print(
        f"device={dev}  batch={vision.size(0)}  T={TINY_SEQ_LEN}  "
        f"dims=({TINY_VISION_DIM}, {TINY_AUDIO_DIM}, {TINY_TEXT_DIM})"
    )
    print(f"{'method':<22} {'out':<18} extra")
    for name, runner in RUNNERS.items():
        fused, extra = runner(vision, audio, text)
        assert torch.isfinite(fused).all(), name
        extra_txt = " ".join(f"{k}={v}" for k, v in extra.items() if k != "fused")
        print(f"{name:<22} {str(extra['fused']):<18} {extra_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
