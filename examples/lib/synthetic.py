"""MOSI/MOSEI-shaped tensors without the CMU SDK or pickle files.

Shapes match `get_ablation_dataloader` / GMTM:

    vision [B, T, 35]
    audio  [B, T, 74]
    text   [B, T, 768]  (BERT) or [B, T, 300] (GloVe)
    label  [B, 1] in [-3, 3]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

VISION_DIM = 35
AUDIO_DIM = 74
BERT_TEXT_DIM = 768
GLOVE_TEXT_DIM = 300
DEFAULT_SEQ_LEN = 50
LABEL_LOW = -3.0
LABEL_HIGH = 3.0

MODALITY_INDEX = {"visual": 0, "audio": 1, "text": 2}


@dataclass(frozen=True)
class FeatureSpec:
    """Feature widths for one text embedding family."""

    name: str
    visual: int = VISION_DIM
    audio: int = AUDIO_DIM
    text: int = BERT_TEXT_DIM
    seq_len: int = DEFAULT_SEQ_LEN

    @property
    def dims(self) -> List[int]:
        return [self.visual, self.audio, self.text]

    @property
    def total_early(self) -> int:
        return self.visual + self.audio + self.text


BERT_SPEC = FeatureSpec("bert", text=BERT_TEXT_DIM)
GLOVE_SPEC = FeatureSpec("glove", text=GLOVE_TEXT_DIM)


@dataclass
class SyntheticBatch:
    """One aligned clip batch plus metadata."""

    vision: np.ndarray
    audio: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    spec: FeatureSpec
    seed: int

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    def as_list(self) -> List[np.ndarray]:
        return [self.vision, self.audio, self.text]

    def shapes(self) -> Dict[str, tuple]:
        return {
            "vision": tuple(self.vision.shape),
            "audio": tuple(self.audio.shape),
            "text": tuple(self.text.shape),
            "labels": tuple(self.labels.shape),
        }


def make_batch(
    batch_size: int = 8,
    spec: FeatureSpec = BERT_SPEC,
    seed: int = 0,
    label_mode: str = "uniform",
) -> SyntheticBatch:
    """Draw Gaussian features and labels in the MOSI/MOSEI range.

    `label_mode`:
      - ``uniform``: labels ~ U[-3, 3]
      - ``mixture``: a 3-component mix (neg / near-zero / pos) so Acc-2 is not trivial
    """
    rng = np.random.default_rng(seed)
    t = spec.seq_len
    vision = rng.normal(0.0, 1.0, size=(batch_size, t, spec.visual)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(batch_size, t, spec.audio)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(batch_size, t, spec.text)).astype(np.float32)

    if label_mode == "uniform":
        labels = rng.uniform(LABEL_LOW, LABEL_HIGH, size=(batch_size, 1)).astype(np.float32)
    elif label_mode == "mixture":
        centers = rng.choice(np.array([-2.0, 0.0, 2.0], dtype=np.float32), size=batch_size)
        noise = rng.normal(0.0, 0.35, size=batch_size).astype(np.float32)
        labels = np.clip(centers + noise, LABEL_LOW, LABEL_HIGH).astype(np.float32).reshape(-1, 1)
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")

    return SyntheticBatch(
        vision=vision, audio=audio, text=text, labels=labels, spec=spec, seed=seed
    )


def ablate_modalities(
    batch: SyntheticBatch,
    keep: Sequence[str],
) -> SyntheticBatch:
    """Zero streams that are not in `keep`, matching `filter_modalities_list`."""
    unknown = set(keep) - set(MODALITY_INDEX)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; expected {sorted(MODALITY_INDEX)}")

    vision = batch.vision if "visual" in keep else np.zeros_like(batch.vision)
    audio = batch.audio if "audio" in keep else np.zeros_like(batch.audio)
    text = batch.text if "text" in keep else np.zeros_like(batch.text)
    return SyntheticBatch(
        vision=vision,
        audio=audio,
        text=text,
        labels=batch.labels.copy(),
        spec=batch.spec,
        seed=batch.seed,
    )


def iter_ablation_sets() -> Iterable[List[str]]:
    """The seven combinations used by `train_GMTM_glove.py`."""
    return (
        ["text"],
        ["audio"],
        ["visual"],
        ["text", "audio"],
        ["text", "visual"],
        ["audio", "visual"],
        ["text", "audio", "visual"],
    )
