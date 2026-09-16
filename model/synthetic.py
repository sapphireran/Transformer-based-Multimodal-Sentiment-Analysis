"""CPU-friendly synthetic stand-ins for CMU-MOSI / CMU-MOSEI batches.

The real pickles are large, licensed academic datasets and are not
checked into this repository. The helpers here invent tensors with the
same ranks and feature widths that ``models.py`` and the training
scripts expect, so documentation examples can run on a laptop without
downloading Facet / COVAREP / BERT features.

Feature widths (per timestep)
-----------------------------
====================  =======  =======
Modality              BERT     GloVe
====================  =======  =======
visual (Facet 4.2)    35       35
audio (COVAREP)       74       74
text                  768      300
====================  =======  =======

Default sequence length is 50, matching ``max_seq_len`` in the
training scripts. Labels are drawn from ``[-3, 3]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

EmbeddingName = Literal["bert", "glove"]

VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_TEXT_DIM = 768
GLOVE_TEXT_DIM = 300
DEFAULT_SEQ_LEN = 50


def text_dim_for(embedding: EmbeddingName) -> int:
    if embedding == "bert":
        return BERT_TEXT_DIM
    if embedding == "glove":
        return GLOVE_TEXT_DIM
    raise ValueError(f"unknown embedding {embedding!r}; expected 'bert' or 'glove'")


def modality_dims(embedding: EmbeddingName = "bert") -> Tuple[int, int, int]:
    """Return ``(visual, audio, text)`` feature widths."""
    return VISUAL_DIM, AUDIO_DIM, text_dim_for(embedding)


@dataclass(frozen=True)
class SyntheticBatch:
    """One packed multimodal batch.

    Attributes
    ----------
    visual, audio, text:
        Float tensors with shape ``[batch, seq_len, feat]``.
    labels:
        Continuous opinion scores with shape ``[batch, 1]``.
    lengths:
        Per-modality valid lengths with shape ``[batch]``. Useful when
        exercising the packed-LSTM fusion setups.
    """

    visual: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor

    def as_list(self) -> List[torch.Tensor]:
        """Order used by ``GatedMultiTransfomerModel``: visual, audio, text."""
        return [self.visual, self.audio, self.text]

    def total_feature_dim(self) -> int:
        return self.visual.size(-1) + self.audio.size(-1) + self.text.size(-1)


def make_synthetic_batch(
    batch_size: int = 8,
    seq_len: int = DEFAULT_SEQ_LEN,
    embedding: EmbeddingName = "bert",
    seed: int = 0,
    device: str | torch.device = "cpu",
    include_signal: bool = True,
) -> SyntheticBatch:
    """Build a reproducible multimodal batch.

    When ``include_signal`` is True, the label is a noisy linear function
    of a few text coordinates so a tiny model can actually fit something.
    When False, features and labels are independent (useful for shape
    tests that should not accidentally "learn").
    """
    rng = np.random.default_rng(seed)
    v_dim, a_dim, t_dim = modality_dims(embedding)

    visual = rng.normal(0.0, 1.0, size=(batch_size, seq_len, v_dim)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(batch_size, seq_len, a_dim)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(batch_size, seq_len, t_dim)).astype(np.float32)

    if include_signal:
        # Mean of the first 8 text dims, scaled into [-3, 3], plus noise.
        proto = text[:, :, :8].mean(axis=(1, 2))
        proto = np.tanh(proto) * 3.0
        noise = rng.normal(0.0, 0.25, size=batch_size)
        labels = np.clip(proto + noise, -3.0, 3.0).astype(np.float32)
    else:
        labels = rng.uniform(-3.0, 3.0, size=batch_size).astype(np.float32)

    lengths = np.full(batch_size, seq_len, dtype=np.int64)

    to_t = lambda array: torch.from_numpy(array).to(device)
    return SyntheticBatch(
        visual=to_t(visual),
        audio=to_t(audio),
        text=to_t(text),
        labels=to_t(labels).unsqueeze(1),
        lengths=torch.from_numpy(lengths).to(device),
    )


def make_synthetic_split(
    n_train: int = 32,
    n_valid: int = 16,
    n_test: int = 16,
    batch_size: int = 8,
    seq_len: int = 16,
    embedding: EmbeddingName = "bert",
    seed: int = 7,
) -> Dict[str, DataLoader]:
    """Return train/valid/test loaders of synthetic clips.

    Sequences are shorter than the real 50-step clips so a laptop CPU
    can finish a couple of GMTM steps in a few seconds.
    """
    train = make_synthetic_batch(n_train, seq_len, embedding, seed=seed)
    valid = make_synthetic_batch(n_valid, seq_len, embedding, seed=seed + 1)
    test = make_synthetic_batch(n_test, seq_len, embedding, seed=seed + 2)

    def _loader(batch: SyntheticBatch, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(batch.visual, batch.audio, batch.text, batch.labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        "train": _loader(train, shuffle=True),
        "valid": _loader(valid, shuffle=False),
        "test": _loader(test, shuffle=False),
    }


def zero_out_modalities(
    batch: SyntheticBatch,
    keep: Sequence[str],
) -> SyntheticBatch:
    """Return a copy that keeps only the named modalities.

    Used by the synthetic ablation example. Names are ``visual``,
    ``audio``, and ``text``. Dropped modalities become zeros, matching
    ``get_ablation_dataloader`` in ``data/get_dataloader.py``.
    """
    allowed = {"visual", "audio", "text"}
    unknown = set(keep) - allowed
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; expected subset of {sorted(allowed)}")
    visual = batch.visual if "visual" in keep else torch.zeros_like(batch.visual)
    audio = batch.audio if "audio" in keep else torch.zeros_like(batch.audio)
    text = batch.text if "text" in keep else torch.zeros_like(batch.text)
    return SyntheticBatch(
        visual=visual,
        audio=audio,
        text=text,
        labels=batch.labels,
        lengths=batch.lengths,
    )
