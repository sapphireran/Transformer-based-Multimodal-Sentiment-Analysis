"""Instantiate GMTM and run a CPU forward pass on a synthetic batch."""

from __future__ import annotations

import torch

from examples.common import (
    BERT_LAYOUT,
    TinyGMTMParams,
    count_parameters,
    cpu_device,
    finite,
    format_shape,
    print_kv,
)
from examples.synthetic_data import make_batch


def build_gmtm(layout=BERT_LAYOUT, hyp_params=TinyGMTMParams):
    from models import GatedMultiTransfomerModel

    return GatedMultiTransfomerModel(3, layout.as_list, hyp_params=hyp_params)


def run_forward(batch_size: int = 4, seq_len: int = 12) -> torch.Tensor:
    device = cpu_device()
    model = build_gmtm().to(device)
    model.eval()
    batch = make_batch(batch_size=batch_size, seq_len=seq_len)
    inputs = [t.to(device) for t in batch.as_list()]
    with torch.no_grad():
        out = model(inputs)
    return model, out, batch


def main() -> int:
    model, out, batch = run_forward()
    print("GMTM CPU forward (tiny HParams: embed=16, layers=1, heads=2)")
    print_kv(
        [
            ("input vision", format_shape(batch.vision)),
            ("input audio", format_shape(batch.audio)),
            ("input text", format_shape(batch.text)),
            ("output", format_shape(out)),
            ("finite", finite(out)),
            ("trainable params", count_parameters(model)),
            ("n_modalities", model.n_modalities),
            ("embed_dim", model.embed_dim),
            ("cross-modal encoders", model.n_modalities**2),
        ]
    )
    if out.shape != (batch.vision.size(0), 1):
        raise RuntimeError(f"expected [B,1], got {tuple(out.shape)}")
    if not finite(out):
        raise RuntimeError("GMTM produced non-finite values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
