"""Tiny CPU training loop for ``GatedMultiTransfomerModel``.

Full MOSI / MOSEI runs need pickles, CUDA, and tens of epochs. This loop
only checks that the gated architecture can overfit a synthetic batch:
the MAE after a handful of AdamW steps should drop below the MAE of the
untrained model. That is enough to keep the examples honest without
pretending to reproduce the CSV tables.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .synthetic import make_sentiment_batch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from models import GatedMultiTransfomerModel  # noqa: E402


class TinyHParams:
    num_heads = 2
    layers = 2
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 8
    attn_mask = False
    output_dim = 1
    all_steps = False


@dataclass
class ToyTrainResult:
    initial_mae: float
    final_mae: float
    history: list[float]
    steps: int
    device: str

    @property
    def improved(self) -> bool:
        return self.final_mae < self.initial_mae


def build_gated_model(feature_dims: list[int], device: torch.device) -> GatedMultiTransfomerModel:
    """GMTM already includes the sentiment head; Identity encoders are implicit."""
    model = GatedMultiTransfomerModel(
        n_modalities=3, n_features=feature_dims, hyp_params=TinyHParams
    )
    return model.to(device)


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred.detach() - target.detach())).cpu())


def train_gated_transformer(
    steps: int = 12,
    batch_size: int = 16,
    seq_len: int = 12,
    lr: float = 8e-3,
    seed: int = 7,
    device: str = "cpu",
) -> ToyTrainResult:
    """Overfit one synthetic batch and return MAE before / after."""
    torch.manual_seed(seed)
    device_t = torch.device(device)
    batch = make_sentiment_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        preset="toy",
        noise=0.05,
        seed=seed,
        device=device_t,
    )
    dims = [batch.dims()[name] for name in ("visual", "audio", "text")]
    model = build_gated_model(dims, device_t)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    model.train()
    history: list[float] = []
    inputs = batch.as_list()
    target = batch.labels

    with torch.no_grad():
        initial = _mae(model(inputs), target)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        final = _mae(model(inputs), target)

    return ToyTrainResult(
        initial_mae=initial,
        final_mae=final,
        history=history,
        steps=steps,
        device=str(device_t),
    )
