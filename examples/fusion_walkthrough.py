"""Forward every fusion primitive on a tiny synthetic batch.

Encoders are scaled down so the script stays CPU-friendly. Printed
``would-be`` widths are the ones from train_main_bert.py so you can match
the docs to the real sweep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import BERT_EARLY_DIM, count_parameters, device, feature_dims
from synthetic_data import make_corpus


def _import_fusion():
    from models import (
        ConcatEarly,
        ConcatLate,
        EarlyFusionTransformer,
        LateFusionTransformer,
        LowRankTensorFusion,
        TensorFusion,
    )

    return (
        ConcatEarly,
        ConcatLate,
        TensorFusion,
        LowRankTensorFusion,
        EarlyFusionTransformer,
        LateFusionTransformer,
    )


def _print(title: str, tensor: torch.Tensor, module: nn.Module) -> None:
    print(f"{title:28s} out={tuple(tensor.shape)}  params={count_parameters(module)}")


def demo_concat(ConcatEarly, ConcatLate, vision, audio, text) -> None:
    early = ConcatEarly()
    y = early([vision, audio, text])
    _print("ConcatEarly", y, early)
    print(f"  {'':28s} expected F={vision.size(-1) + audio.size(-1) + text.size(-1)}")

    # Late concat expects already-pooled vectors; flatten each (B, T, F).
    late = ConcatLate()
    y = late([vision, audio, text])
    _print("ConcatLate (flatten T*F)", y, late)
    print(f"  {'':28s} BERT late-LSTM width after encode is 64+256+1024=1344")


def demo_tfn(TensorFusion) -> None:
    # Tiny stand-in for the BERT GRUWithLinear outputs 19 / 39 / 159.
    b = 2
    mods = [
        torch.randn(b, 4, device=device()),
        torch.randn(b, 5, device=device()),
        torch.randn(b, 6, device=device()),
    ]
    tfn = TensorFusion()
    y = tfn(mods)
    _print("TensorFusion toy 4x5x6", y, tfn)
    print(f"  {'':28s} with homogeneous 1s: (4+1)*(5+1)*(6+1) = {5 * 6 * 7}")
    print(f"  {'':28s} BERT sweep uses 20*40*160 = 128000 then MLP(128000, 2048, 1)")


def demo_lmf(LowRankTensorFusion) -> None:
    b = 2
    dims = (8, 16, 32)
    mods = [torch.randn(b, d, device=device()) for d in dims]
    lmf = LowRankTensorFusion(list(dims), output_dim=16, rank=4).to(device())
    y = lmf(mods)
    _print("LowRankTensorFusion toy", y, lmf)
    print(f"  {'':28s} BERT sweep: inputs [32, 64, 256], out=256, rank=32")


def demo_transformers(
    EarlyFusionTransformer,
    LateFusionTransformer,
    vision,
    audio,
    text,
) -> None:
    early = EarlyFusionTransformer(n_features=BERT_EARLY_DIM).to(device())
    y = early([vision, audio, text])
    _print("EarlyFusionTransformer", y, early)
    print(f"  {'':28s} last timestep, embed_dim=32 (head should be 32-D, not 64-D)")

    # Fake per-modality transformer outputs at the BERT late-fusion widths.
    b, t = vision.shape[:2]
    encoded = [
        torch.randn(b, t, 64, device=device()),
        torch.randn(b, t, 128, device=device()),
        torch.randn(b, t, 1024, device=device()),
    ]
    late = LateFusionTransformer(in_dim=64 + 128 + 1024).to(device())
    y = late(encoded)
    _print("LateFusionTransformer", y, late)
    print(f"  {'':28s} in_dim=1216, last timestep embed_dim=32")


def main() -> None:
    torch.manual_seed(5)
    (
        ConcatEarly,
        ConcatLate,
        TensorFusion,
        LowRankTensorFusion,
        EarlyFusionTransformer,
        LateFusionTransformer,
    ) = _import_fusion()

    corpus = make_corpus(n=2, text="bert", min_len=6, max_len=10, seed=5)
    vision, audio, text, _ = corpus.padded(max_len=10)
    fv, fa, ft = feature_dims("bert")
    print(f"device: {device()}")
    print(f"padded batch: vision {tuple(vision.shape)} audio {tuple(audio.shape)} text {tuple(text.shape)}")
    print(f"BERT early-concat F = {fv}+{fa}+{ft} = {BERT_EARLY_DIM}")
    print()
    demo_concat(ConcatEarly, ConcatLate, vision, audio, text)
    print()
    demo_tfn(TensorFusion)
    print()
    demo_lmf(LowRankTensorFusion)
    print()
    demo_transformers(EarlyFusionTransformer, LateFusionTransformer, vision, audio, text)


if __name__ == "__main__":
    main()
