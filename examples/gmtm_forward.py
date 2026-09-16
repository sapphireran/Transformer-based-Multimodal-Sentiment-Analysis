"""Forward-pass demo of ``GatedMultiTransfomerModel`` on CPU.

Uses a shrunk ``HParams`` (embed 16, one layer, two heads) so the
pairwise graph still exists — 3×3 encoders — without the 64-D / 4-layer
study configuration.
"""

from __future__ import annotations

import torch

import _paths  # noqa: F401  — puts model/ on sys.path
from models import GatedMultiTransfomerModel
from synthetic_batch import make_synthetic_batch, zero_modalities


class TinyHParams:
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


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_tiny_gmtm(
    n_features: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> GatedMultiTransfomerModel:
    if n_features is None:
        n_features = [8, 10, 12]
    model = GatedMultiTransfomerModel(
        n_modalities=3, n_features=n_features, hyp_params=TinyHParams
    )
    return model.to(device).eval()


@torch.no_grad()
def run_forward(seed: int = 0) -> dict[str, torch.Tensor]:
    n_features = [8, 10, 12]
    model = build_tiny_gmtm(n_features)
    batch = make_synthetic_batch(
        batch_size=4,
        seq_len=10,
        visual_dim=n_features[0],
        audio_dim=n_features[1],
        text_dim=n_features[2],
        seed=seed,
    )
    full = model(batch.as_gmtm_input())
    text_only = model(zero_modalities(batch, keep=("text",)).as_gmtm_input())
    return {
        "full": full,
        "text_only": text_only,
        "labels": batch.labels,
        "n_params": torch.tensor(count_parameters(model)),
    }


def _demo() -> None:
    n_features = [8, 10, 12]
    model = build_tiny_gmtm(n_features)
    batch = make_synthetic_batch(
        batch_size=4,
        seq_len=10,
        visual_dim=8,
        audio_dim=10,
        text_dim=12,
    )
    print("tiny GMTM")
    print(f"  n_features={n_features} embed_dim={TinyHParams.embed_dim} "
          f"layers={TinyHParams.layers} heads={TinyHParams.num_heads}")
    print(f"  pairwise encoders: {len(model.trans)} x {len(model.trans[0])}")
    print(f"  parameters: {count_parameters(model)}")
    out = model(batch.as_gmtm_input())
    print(f"  forward {tuple(out.shape)}  "
          f"values={out.squeeze(-1).detach().tolist()}")
    masked = model(zero_modalities(batch, keep=("text",)).as_gmtm_input())
    delta = (out - masked).abs().mean().item()
    print(f"  |full - text-only| mean {delta:.4f} "
          "(zeroing vision/audio changes the prediction, as expected)")
    weights = torch.softmax(model.modal_weights, dim=0).detach().tolist()
    print(f"  softmax(modal_weights) at init ≈ {['%.3f' % w for w in weights]}")


if __name__ == "__main__":
    _demo()
