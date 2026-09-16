"""Synthetic tensors with the same ranks as the MOSI / MOSEI loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .repo import FEATURE_WIDTHS, PADDED_SEQ_LEN, SENTIMENT_HIGH, SENTIMENT_LOW

EmbeddingName = Literal["bert", "glove"]


@dataclass(frozen=True)
class ToyMixtureSpec:
    """Linear mixture used to build learnable synthetic labels."""

    visual_weight: float = -0.25
    audio_weight: float = 0.35
    text_weight: float = 1.40
    noise_std: float = 0.15


def feature_widths(embedding: EmbeddingName = "bert") -> Dict[str, int]:
    return dict(FEATURE_WIDTHS[embedding])


def make_mosei_like_batch(
    batch_size: int = 4,
    seq_len: int = PADDED_SEQ_LEN,
    embedding: EmbeddingName = "bert",
    seed: int = 0,
    device: Optional[object] = None,
):
    """Return (vision, audio, text, labels) as torch tensors.

    Labels are random Uniform[-3, 3], independent of the features. Use
    :func:`make_toy_mixture` when you want a learnable target.
    """
    import torch

    rng = np.random.default_rng(seed)
    widths = feature_widths(embedding)
    vision = rng.normal(0.0, 1.0, size=(batch_size, seq_len, widths["visual"]))
    audio = rng.normal(0.0, 1.0, size=(batch_size, seq_len, widths["audio"]))
    text = rng.normal(0.0, 1.0, size=(batch_size, seq_len, widths["text"]))
    # First text frame is never all zeros so Affectdataset-style alignment works.
    text[:, 0, 0] = rng.choice([-1.0, 1.0], size=batch_size)
    labels = rng.uniform(SENTIMENT_LOW, SENTIMENT_HIGH, size=(batch_size, 1))

    tensors = [
        torch.tensor(vision, dtype=torch.float32),
        torch.tensor(audio, dtype=torch.float32),
        torch.tensor(text, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    ]
    if device is not None:
        tensors = [item.to(device) for item in tensors]
    return tuple(tensors)


def _pool_mean(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=(1, 2))


def make_toy_mixture(
    batch_size: int = 16,
    seq_len: int = 8,
    n_features: Sequence[int] = (8, 12, 16),
    spec: ToyMixtureSpec = ToyMixtureSpec(),
    seed: int = 7,
    device: Optional[object] = None,
) -> Tuple[List[object], object]:
    """Build a small 3-modality batch whose label is a linear mixture.

    `n_features` is (visual, audio, text) to match GMTM's slot order.
    The target is affine in the three mean-pooled streams, clipped to
    [-3, 3], plus a little noise. A working GMTM should drive MAE down.
    """
    import torch

    if len(n_features) != 3:
        raise ValueError("toy mixture expects three modality widths")

    rng = np.random.default_rng(seed)
    streams = [
        rng.normal(0.0, 1.0, size=(batch_size, seq_len, int(width)))
        for width in n_features
    ]
    raw = (
        spec.visual_weight * _pool_mean(streams[0])
        + spec.audio_weight * _pool_mean(streams[1])
        + spec.text_weight * _pool_mean(streams[2])
        + rng.normal(0.0, spec.noise_std, size=(batch_size,))
    )
    # Stretch to the sentiment range so Acc7 bins are actually populated.
    scaled = SENTIMENT_LOW + (SENTIMENT_HIGH - SENTIMENT_LOW) * (
        (raw - raw.min()) / ((raw.max() - raw.min()) + 1e-6)
    )
    labels = scaled.reshape(batch_size, 1).astype(np.float32)

    tensors = [torch.tensor(stream, dtype=torch.float32) for stream in streams]
    target = torch.tensor(labels, dtype=torch.float32)
    if device is not None:
        tensors = [item.to(device) for item in tensors]
        target = target.to(device)
    return tensors, target


def make_tiny_affect_dict(
    n_train: int = 6,
    n_valid: int = 4,
    n_test: int = 4,
    seq_len: int = PADDED_SEQ_LEN,
    embedding: EmbeddingName = "bert",
    seed: int = 1,
) -> Dict[str, Dict[str, object]]:
    """Pickle-compatible nested dict: split → {vision, audio, text, labels, id}."""

    rng = np.random.default_rng(seed)
    widths = feature_widths(embedding)

    def _split(name: str, count: int, split_seed: int) -> Dict[str, object]:
        local = np.random.default_rng(split_seed)
        vision = local.normal(0.0, 1.0, size=(count, seq_len, widths["visual"])).astype(
            np.float32
        )
        audio = local.normal(0.0, 1.0, size=(count, seq_len, widths["audio"])).astype(
            np.float32
        )
        text = local.normal(0.0, 1.0, size=(count, seq_len, widths["text"])).astype(
            np.float32
        )
        text[:, 0, 0] = 1.0
        labels = local.uniform(SENTIMENT_LOW, SENTIMENT_HIGH, size=(count, 1, 1)).astype(
            np.float32
        )
        ids = [f"synth_{name}_{i}" for i in range(count)]
        return {
            "vision": vision,
            "audio": audio,
            "text": text,
            "labels": labels,
            "id": ids,
        }

    # Consume rng so successive calls with different seeds stay independent.
    _ = rng.integers(0, 10_000, size=3)
    return {
        "train": _split("train", n_train, seed + 10),
        "valid": _split("valid", n_valid, seed + 20),
        "test": _split("test", n_test, seed + 30),
    }
