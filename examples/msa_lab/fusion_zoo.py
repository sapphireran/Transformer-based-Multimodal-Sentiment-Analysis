"""CPU-friendly fusion zoo that reuses the modules in ``model/models.py``.

Real training scripts pair encoders with fusion and a head, then wrap
everything in ``MultiFramework``. The examples only need forward-pass
shapes, so each zoo entry stores:

* the fusion module
* a short description
* a ``prepare(batch) -> fused`` callable that knows how that fusion
  wants its inputs (list of ``[B, F]`` vectors vs ``[B, T, F]`` sequences)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .synthetic import SentimentBatch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from models import (  # noqa: E402
    ConcatEarly,
    ConcatLate,
    GatedMultiTransfomerModel,
    LowRankTensorFusion,
    TensorFusion,
    TransformerFusion,
)


@dataclass
class FusionSpec:
    name: str
    family: str
    module: nn.Module
    prepare: Callable[[SentimentBatch], torch.Tensor]
    notes: str

    def fused_shape(self, batch: SentimentBatch) -> tuple[int, ...]:
        self.module.eval()
        with torch.no_grad():
            fused = self.prepare(batch)
        return tuple(int(dim) for dim in fused.shape)


def _mean_pool(batch: SentimentBatch) -> list[torch.Tensor]:
    return [tensor.mean(dim=1) for tensor in batch.as_list()]


def _concat_early(fusion: ConcatEarly) -> Callable[[SentimentBatch], torch.Tensor]:
    def prepare(batch: SentimentBatch) -> torch.Tensor:
        return fusion(batch.as_list())

    return prepare


def _concat_late(fusion: ConcatLate) -> Callable[[SentimentBatch], torch.Tensor]:
    def prepare(batch: SentimentBatch) -> torch.Tensor:
        return fusion(_mean_pool(batch))

    return prepare


def _tensor_fusion(fusion: TensorFusion) -> Callable[[SentimentBatch], torch.Tensor]:
    def prepare(batch: SentimentBatch) -> torch.Tensor:
        return fusion(_mean_pool(batch))

    return prepare


def _low_rank(fusion: LowRankTensorFusion) -> Callable[[SentimentBatch], torch.Tensor]:
    def prepare(batch: SentimentBatch) -> torch.Tensor:
        return fusion(_mean_pool(batch))

    return prepare


class _HParams:
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


def _gated(fusion: GatedMultiTransfomerModel) -> Callable[[SentimentBatch], torch.Tensor]:
    def prepare(batch: SentimentBatch) -> torch.Tensor:
        return fusion(batch.as_list())

    return prepare


def build_fusion_zoo(feature_dims: dict[str, int] | None = None) -> list[FusionSpec]:
    """Construct a CPU zoo for the given per-modality feature sizes.

    Default sizes match the ``toy`` synthetic preset (8 / 12 / 16) so the
    Tensor Fusion outer product stays small enough for laptops.
    """
    dims = feature_dims or {"visual": 8, "audio": 12, "text": 16}
    visual, audio, text = dims["visual"], dims["audio"], dims["text"]
    concat_dim = visual + audio + text
    tensor_out = (visual + 1) * (audio + 1) * (text + 1)

    concat_early = ConcatEarly()
    concat_late = ConcatLate()
    tensor_fusion = TensorFusion()
    low_rank = LowRankTensorFusion([visual, audio, text], output_dim=16, rank=4)
    # TransformerFusion stacks modality vectors; they must share d_model.
    shared = 16
    transformer_fusion = TransformerFusion(d_model=shared, nhead=4, num_layers=1, dropout=0.0)
    gated = GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[visual, audio, text],
        hyp_params=_HParams,
    )

    def prepare_transformer(batch: SentimentBatch) -> torch.Tensor:
        pooled = _mean_pool(batch)
        projected = []
        for tensor in pooled:
            if tensor.size(-1) == shared:
                projected.append(tensor)
            elif tensor.size(-1) > shared:
                projected.append(tensor[:, :shared])
            else:
                projected.append(
                    torch.nn.functional.pad(tensor, (0, shared - tensor.size(-1)))
                )
        return transformer_fusion(projected)

    zoo = [
        FusionSpec(
            name="ConcatEarly",
            family="concat",
            module=concat_early,
            prepare=_concat_early(concat_early),
            notes=(
                f"Cat along the feature axis: [B, T, {concat_dim}]. "
                "Training then runs an LSTM on the packed sequence."
            ),
        ),
        FusionSpec(
            name="ConcatLate",
            family="concat",
            module=concat_late,
            prepare=_concat_late(concat_late),
            notes=(
                f"Mean-pool each stream then cat: [B, {concat_dim}]. "
                "Training uses LSTM/GRU encoders instead of mean-pool."
            ),
        ),
        FusionSpec(
            name="TensorFusion",
            family="tensor",
            module=tensor_fusion,
            prepare=_tensor_fusion(tensor_fusion),
            notes=(
                f"Outer product of bias-augmented vectors -> [B, {tensor_out}]. "
                "MOSEI BERT training uses 19 x 39 x 159 = 128000 features."
            ),
        ),
        FusionSpec(
            name="LowRankTensorFusion",
            family="tensor",
            module=low_rank,
            prepare=_low_rank(low_rank),
            notes="Rank-4 factors map the same outer-product idea to 16-d.",
        ),
        FusionSpec(
            name="TransformerFusion",
            family="transformer",
            module=transformer_fusion,
            prepare=prepare_transformer,
            notes="Treat the three pooled modalities as a length-3 token sequence.",
        ),
        FusionSpec(
            name="GatedMultiTransformer",
            family="transformer",
            module=gated,
            prepare=_gated(gated),
            notes=(
                "Pairwise cross-modal transformers, learned modality weights, "
                "sigmoid gates, attention pooling, then a 1-d sentiment head."
            ),
        ),
    ]
    for spec in zoo:
        spec.module.eval()
        for parameter in spec.module.parameters():
            parameter.requires_grad_(False)
    return zoo


def describe_fusion_shapes(
    zoo: list[FusionSpec], batch: SentimentBatch
) -> list[dict[str, str]]:
    rows = []
    for spec in zoo:
        shape = spec.fused_shape(batch)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "shape": "x".join(str(dim) for dim in shape),
                "notes": spec.notes,
            }
        )
    return rows
