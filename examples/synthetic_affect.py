"""Synthetic MOSI/MOSEI-shaped batches that do not need the official pickles.

The loaders in ``model/data/get_dataloader.py`` expect four keys per split:
``vision`` (35), ``audio`` (74), ``text`` (768 or 300), ``labels``. This module
builds the same layout in memory, plus a *learnable* label so GMTM can train
without downloading CMU data.

Labels are a bounded function of the text stream, with a little audio/vision
leak and Gaussian noise. That matches the ablation story in
``docs/ablation-study.md``: text is the useful modality; the others are weak
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence

import numpy as np
import torch

Embedding = Literal["bert", "glove"]

VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_DIM = 768
GLOVE_DIM = 300
MAX_LEN = 50
LABEL_LOW = -3.0
LABEL_HIGH = 3.0


def text_dim(embedding: Embedding) -> int:
    return BERT_DIM if embedding == "bert" else GLOVE_DIM


@dataclass(frozen=True)
class AffectShapes:
    """Official feature widths used throughout ``model/``."""

    visual: int = VISUAL_DIM
    audio: int = AUDIO_DIM
    text: int = BERT_DIM
    max_len: int = MAX_LEN
    embedding: Embedding = "bert"

    @classmethod
    def for_embedding(cls, embedding: Embedding, max_len: int = MAX_LEN) -> "AffectShapes":
        return cls(
            visual=VISUAL_DIM,
            audio=AUDIO_DIM,
            text=text_dim(embedding),
            max_len=max_len,
            embedding=embedding,
        )


def _unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / (norm if norm > 0 else 1.0)


def _projected_score(arr: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Time-mean then unit-project, scaled to roughly N(0, 1)."""
    pooled = arr.mean(axis=1)
    return (pooled @ direction) * np.sqrt(arr.shape[1])


def make_split_arrays(
    n: int,
    shapes: AffectShapes | None = None,
    seed: int = 0,
    text_weight: float = 1.0,
    audio_weight: float = 0.15,
    visual_weight: float = 0.10,
    noise: float = 0.10,
) -> Dict[str, np.ndarray]:
    """Return a dict of numpy arrays with the pickle-split schema.

    ``labels`` has shape ``(N, 1)``. Sequences have shape ``(N, T, F)``.
    Sentiment is a bounded function of a fixed projection of the text
    stream, plus weak audio/vision leaks — the same qualitative story as
    the BERT ablation table.
    """
    shapes = shapes or AffectShapes()
    rng = np.random.default_rng(seed)
    t, n_ = shapes.max_len, n

    vis_dir = _unit_vector(rng, shapes.visual)
    aud_dir = _unit_vector(rng, shapes.audio)
    txt_dir = _unit_vector(rng, shapes.text)

    vision = rng.normal(0.0, 1.0, size=(n_, t, shapes.visual)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(n_, t, shapes.audio)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(n_, t, shapes.text)).astype(np.float32)

    raw = (
        text_weight * _projected_score(text, txt_dir)
        + audio_weight * _projected_score(audio, aud_dir)
        + visual_weight * _projected_score(vision, vis_dir)
        + rng.normal(0.0, noise, size=(n_,))
    )
    # squash to the MOSI/MOSEI Likert-ish range
    labels = np.tanh(raw) * LABEL_HIGH
    labels = labels.astype(np.float32).reshape(n_, 1)

    ids = np.array([f"synth[{i}]" for i in range(n_)])
    return {
        "vision": vision,
        "audio": audio,
        "text": text,
        "labels": labels,
        "id": ids,
        "_text_dir": txt_dir,
        "_audio_dir": aud_dir,
        "_vision_dir": vis_dir,
    }


def make_pickle_dict(
    n_train: int = 32,
    n_valid: int = 16,
    n_test: int = 16,
    shapes: AffectShapes | None = None,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """In-memory stand-in for ``mosei_raw_bert.pkl`` / ``mosi_raw_glove.pkl``."""
    shapes = shapes or AffectShapes()
    return {
        "train": make_split_arrays(n_train, shapes, seed=seed),
        "valid": make_split_arrays(n_valid, shapes, seed=seed + 1),
        "test": make_split_arrays(n_test, shapes, seed=seed + 2),
    }


def arrays_to_tensors(
    split: Dict[str, np.ndarray],
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    device = device or torch.device("cpu")
    out: Dict[str, torch.Tensor] = {}
    for key in ("vision", "audio", "text", "labels"):
        out[key] = torch.from_numpy(np.asarray(split[key])).to(device)
    return out


def batch_from_split(
    split: Dict[str, np.ndarray],
    batch_size: int,
    offset: int = 0,
    device: torch.device | None = None,
) -> List[torch.Tensor]:
    """Return ``[vision, audio, text, labels]`` like a ``max_pad`` collate."""
    tensors = arrays_to_tensors(split, device=device)
    sl = slice(offset, offset + batch_size)
    return [tensors["vision"][sl], tensors["audio"][sl], tensors["text"][sl], tensors["labels"][sl]]


def zero_unused_modalities(
    vision: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    keep: Sequence[str],
) -> List[torch.Tensor]:
    """Mirror ``get_ablation_dataloader``: keep listed streams, zero the rest."""
    keep_set = {name.lower() for name in keep}
    if "visual" not in keep_set and "vision" not in keep_set:
        vision = torch.zeros_like(vision)
    if "audio" not in keep_set:
        audio = torch.zeros_like(audio)
    if "text" not in keep_set:
        text = torch.zeros_like(text)
    return [vision, audio, text]


def iter_maxpad_batches(
    split: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device | None = None,
) -> Iterable[List[torch.Tensor]]:
    n = split["labels"].shape[0]
    for offset in range(0, n, batch_size):
        yield batch_from_split(split, batch_size, offset=offset, device=device)
