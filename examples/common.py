"""Shared helpers for the personal CPU examples.

The research code lives under ``model/`` and expects that directory on
``sys.path`` (the train scripts do ``sys.path.append(os.getcwd())`` after
``cd model``). Examples keep the same convention without requiring a cd.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_RESULTS_DIR = MODEL_DIR / "mosi_test"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_DIM = 768
GLOVE_DIM = 300
SEQ_LEN = 50


@dataclass(frozen=True)
class SyntheticBatch:
    """One packed-looking multimodal batch with a scalar label."""

    vision: torch.Tensor  # [B, T, 35]
    audio: torch.Tensor  # [B, T, 74]
    text: torch.Tensor  # [B, T, Dt]
    labels: torch.Tensor  # [B, 1]
    embedding: str

    @property
    def as_list(self) -> List[torch.Tensor]:
        return [self.vision, self.audio, self.text]

    @property
    def text_dim(self) -> int:
        return self.text.size(-1)

    @property
    def early_width(self) -> int:
        return VISUAL_DIM + AUDIO_DIM + self.text_dim


def make_synthetic_batch(
    batch_size: int = 4,
    seq_len: int = SEQ_LEN,
    embedding: str = "bert",
    seed: int = 0,
) -> SyntheticBatch:
    """Build MOSI/MOSEI-shaped tensors that do not come from CMU data.

    Values are standard-normal features and labels clipped to [-3, 3], which
    is the same numeric range ``single_test`` assumes.
    """

    if embedding not in {"bert", "glove"}:
        raise ValueError(f"embedding must be 'bert' or 'glove', got {embedding!r}")

    text_dim = BERT_DIM if embedding == "bert" else GLOVE_DIM
    g = torch.Generator().manual_seed(seed)
    vision = torch.randn(batch_size, seq_len, VISUAL_DIM, generator=g)
    audio = torch.randn(batch_size, seq_len, AUDIO_DIM, generator=g)
    text = torch.randn(batch_size, seq_len, text_dim, generator=g)
    labels = torch.clamp(torch.randn(batch_size, 1, generator=g) * 1.5, -3.0, 3.0)
    return SyntheticBatch(vision, audio, text, labels, embedding)


def zero_modalities(batch: SyntheticBatch, keep: Tuple[str, ...]) -> List[torch.Tensor]:
    """Match ``get_ablation_dataloader``: keep named streams, zero the rest."""

    allowed = {"visual", "audio", "text"}
    unknown = set(keep) - allowed
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}")

    streams = {
        "visual": batch.vision.clone(),
        "audio": batch.audio.clone(),
        "text": batch.text.clone(),
    }
    for name, tensor in streams.items():
        if name not in keep:
            streams[name] = torch.zeros_like(tensor)
    return [streams["visual"], streams["audio"], streams["text"]]


def pool_last(modalities: List[torch.Tensor]) -> List[torch.Tensor]:
    """Cheap stand-in for an LSTM/GRU encoder: last time step."""

    return [m[:, -1, :] for m in modalities]


def tensor_stats(name: str, tensor: torch.Tensor) -> Dict[str, float]:
    t = tensor.detach().float().reshape(-1)
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
        "finite": bool(torch.isfinite(t).all()),
    }


def format_stats(stats: Dict[str, float]) -> str:
    return (
        f"{stats['name']:28s} shape={stats['shape']!s:20s} "
        f"mean={stats['mean']:+.4f} std={stats['std']:.4f} finite={stats['finite']}"
    )
