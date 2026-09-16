"""Synthetic MOSI / MOSEI-shaped batches for CPU examples.

The real pickles are not in git. This module builds in-memory tensors with
the same feature widths the training scripts expect, plus a label that is a
noisy function of the text stream so a tiny GMTM can actually descend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

VISION_DIM = 35
AUDIO_DIM = 74
BERT_DIM = 768
GLOVE_DIM = 300
DEFAULT_SEQ_LEN = 50

TEXT_BACKENDS = {
    "bert": BERT_DIM,
    "glove": GLOVE_DIM,
}


@dataclass(frozen=True)
class ToyBatch:
    """One padded batch in the layout ``_process_2`` produces."""

    vision: np.ndarray  # [B, T, 35]
    audio: np.ndarray  # [B, T, 74]
    text: np.ndarray  # [B, T, 768|300]
    labels: np.ndarray  # [B, 1]
    text_backend: str
    modalities: Tuple[str, ...] = ("visual", "audio", "text")

    @property
    def batch_size(self) -> int:
        return int(self.vision.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.vision.shape[1])

    def as_modality_list(self) -> List[np.ndarray]:
        return [self.vision, self.audio, self.text]

    def masked(self, keep: Sequence[str]) -> "ToyBatch":
        """Zero dropped modalities, matching ``get_ablation_dataloader``."""
        keep_set = set(keep)
        vision = self.vision if "visual" in keep_set else np.zeros_like(self.vision)
        audio = self.audio if "audio" in keep_set else np.zeros_like(self.audio)
        text = self.text if "text" in keep_set else np.zeros_like(self.text)
        return ToyBatch(
            vision=vision,
            audio=audio,
            text=text,
            labels=self.labels.copy(),
            text_backend=self.text_backend,
            modalities=tuple(keep),
        )


def _require_backend(text_backend: str) -> int:
    try:
        return TEXT_BACKENDS[text_backend]
    except KeyError as exc:
        raise ValueError(
            f"text_backend must be one of {sorted(TEXT_BACKENDS)}, got {text_backend!r}"
        ) from exc


def make_toy_batch(
    batch_size: int = 8,
    seq_len: int = DEFAULT_SEQ_LEN,
    text_backend: str = "bert",
    seed: int = 0,
    label_noise: float = 0.15,
) -> ToyBatch:
    """Build one aligned multimodal batch with labels in ``[-3, 3]``.

    The latent sentiment is a weighted mix of the three modality means, with
    text carrying most of the mass (the same qualitative pattern the MOSEI
    ablation tables show). Noise keeps Acc-7 below 1.0 so the metric helpers
    have something non-trivial to report.
    """
    text_dim = _require_backend(text_backend)
    rng = np.random.default_rng(seed)

    vision = rng.normal(0.0, 1.0, size=(batch_size, seq_len, VISION_DIM)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(batch_size, seq_len, AUDIO_DIM)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(batch_size, seq_len, text_dim)).astype(np.float32)

    vision_score = vision.mean(axis=(1, 2))
    audio_score = audio.mean(axis=(1, 2))
    text_score = text.mean(axis=(1, 2))
    # Scale the almost-zero means so tanh is not stuck near 0.
    latent = 8.0 * (0.15 * vision_score + 0.15 * audio_score + 0.70 * text_score)
    labels = np.tanh(latent) * 3.0
    labels = labels + rng.normal(0.0, label_noise, size=batch_size)
    labels = np.clip(labels, -3.0, 3.0).astype(np.float32).reshape(batch_size, 1)

    return ToyBatch(
        vision=vision,
        audio=audio,
        text=text,
        labels=labels,
        text_backend=text_backend,
    )


def make_toy_split(
    n_train: int = 64,
    n_valid: int = 16,
    n_test: int = 16,
    batch_size: int = 8,
    seq_len: int = 16,
    text_backend: str = "bert",
    seed: int = 0,
) -> Dict[str, List[ToyBatch]]:
    """Return train/valid/test lists of ``ToyBatch`` objects."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    counts = {"train": n_train, "valid": n_valid, "test": n_test}
    splits: Dict[str, List[ToyBatch]] = {}
    cursor = seed
    for name, n_samples in counts.items():
        batches = []
        remaining = n_samples
        while remaining > 0:
            this_bs = min(batch_size, remaining)
            batches.append(
                make_toy_batch(
                    batch_size=this_bs,
                    seq_len=seq_len,
                    text_backend=text_backend,
                    seed=cursor,
                )
            )
            cursor += 1
            remaining -= this_bs
        splits[name] = batches
    return splits


def iter_torch_batches(
    batches: Iterable[ToyBatch],
    device: str = "cpu",
):
    """Yield ``(vision, audio, text, labels)`` tensors from toy batches."""
    import torch

    torch_device = torch.device(device)
    for batch in batches:
        yield (
            torch.from_numpy(batch.vision).to(torch_device),
            torch.from_numpy(batch.audio).to(torch_device),
            torch.from_numpy(batch.text).to(torch_device),
            torch.from_numpy(batch.labels).to(torch_device),
        )


def describe_batch(batch: ToyBatch) -> str:
    return (
        f"backend={batch.text_backend}  "
        f"B={batch.batch_size} T={batch.seq_len}  "
        f"vision{tuple(batch.vision.shape)}  "
        f"audio{tuple(batch.audio.shape)}  "
        f"text{tuple(batch.text.shape)}  "
        f"labels[{batch.labels.min():.2f}, {batch.labels.max():.2f}]"
    )
