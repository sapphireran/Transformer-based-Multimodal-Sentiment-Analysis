"""Synthetic MOSI / MOSEI-shaped batches for local examples.

The real loaders read large pickles. These helpers only match ranks,
dtypes, and the two collate signatures in ``model/data/get_dataloader.py``
so the fusion modules and GMTM can run without CMU-SDK files.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"


def ensure_model_on_path() -> Path:
    """Make ``import models`` and ``import train_and_test`` work."""
    path = str(MODEL_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return MODEL_DIR


ensure_model_on_path()


# Official feature widths used by the recorded scripts.
BERT_DIMS = {"visual": 35, "audio": 74, "text": 768}
GLOVE_DIMS = {"visual": 35, "audio": 74, "text": 300}

# Tiny widths so CPU examples finish in seconds.
TOY_DIMS = {"visual": 8, "audio": 10, "text": 16}


@dataclass(frozen=True)
class FeatureSpec:
    """Per-stream widths and default sequence length."""

    visual: int = TOY_DIMS["visual"]
    audio: int = TOY_DIMS["audio"]
    text: int = TOY_DIMS["text"]
    seq_len: int = 12

    @property
    def as_list(self) -> List[int]:
        return [self.visual, self.audio, self.text]

    @property
    def total(self) -> int:
        return self.visual + self.audio + self.text


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 7) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split(
    n: int,
    spec: FeatureSpec = FeatureSpec(),
    seed: int = 0,
    label_from_text: bool = True,
) -> Dict[str, np.ndarray]:
    """Build one ``{'vision','audio','text','labels'}`` split.

    Labels sit in ``[-3, 3]``. When ``label_from_text`` is true they
    are a noisy function of the text mean so a tiny GMTM can drive
    training loss down. Otherwise they are uniform noise.
    """
    rng = np.random.default_rng(seed)
    vision = rng.normal(0.0, 1.0, size=(n, spec.seq_len, spec.visual)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(n, spec.seq_len, spec.audio)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(n, spec.seq_len, spec.text)).astype(np.float32)

    # Leave the first row of a few clips non-zero so an "aligned"
    # slice still has a start index, matching Affectdataset.
    text[:, 0, :] += 0.25

    if label_from_text:
        raw = 2.5 * np.tanh(text.mean(axis=(1, 2)))
        raw = raw + 0.15 * rng.normal(0.0, 1.0, size=n)
    else:
        raw = rng.uniform(-3.0, 3.0, size=n)
    labels = np.clip(raw, -3.0, 3.0).astype(np.float32).reshape(n, 1)

    return {
        "vision": vision,
        "audio": audio,
        "text": text,
        "labels": labels,
    }


def make_dataset(
    n_train: int = 32,
    n_valid: int = 16,
    n_test: int = 16,
    spec: FeatureSpec = FeatureSpec(),
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "train": make_split(n_train, spec, seed=seed),
        "valid": make_split(n_valid, spec, seed=seed + 1),
        "test": make_split(n_test, spec, seed=seed + 2),
    }


def tensors_from_split(
    split: Dict[str, np.ndarray],
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batch-first ``(vision, audio, text, labels)`` tensors."""
    device = device or pick_device()
    vision = torch.from_numpy(split["vision"]).to(device)
    audio = torch.from_numpy(split["audio"]).to(device)
    text = torch.from_numpy(split["text"]).to(device)
    labels = torch.from_numpy(split["labels"]).to(device)
    return vision, audio, text, labels


def random_padded_batch(
    batch_size: int = 8,
    spec: FeatureSpec = FeatureSpec(),
    device: torch.device | None = None,
    seed: int = 11,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """``_process_2``-style batch: three ``(B, T, F)`` streams + labels."""
    device = device or pick_device()
    split = make_split(batch_size, spec, seed=seed)
    vision, audio, text, labels = tensors_from_split(split, device)
    return [vision, audio, text], labels


def random_variable_lengths(
    batch_size: int = 8,
    spec: FeatureSpec = FeatureSpec(),
    min_len: int = 4,
    seed: int = 11,
) -> List[List[torch.Tensor]]:
    """Per-sample lists ``[vision, audio, text, index, label]``.

    Lengths differ so the packed collate has something to pad.
    """
    rng = np.random.default_rng(seed)
    rows: List[List[torch.Tensor]] = []
    for i in range(batch_size):
        t = int(rng.integers(min_len, spec.seq_len + 1))
        vision = torch.from_numpy(
            rng.normal(0.0, 1.0, size=(t, spec.visual)).astype(np.float32)
        )
        audio = torch.from_numpy(
            rng.normal(0.0, 1.0, size=(t, spec.audio)).astype(np.float32)
        )
        text = torch.from_numpy(
            rng.normal(0.0, 1.0, size=(t, spec.text)).astype(np.float32)
        )
        label = torch.tensor([[float(rng.uniform(-3.0, 3.0))]], dtype=torch.float32)
        rows.append([vision, audio, text, torch.tensor(i), label])
    return rows


def collate_packed(
    rows: Sequence[Sequence[torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Stand-in for ``_process_1``.

    Returns ``(features, lengths, indices, labels)``.
    """
    from torch.nn.utils.rnn import pad_sequence

    features: List[torch.Tensor] = []
    lengths: List[torch.Tensor] = []
    for i in range(3):
        seqs = [row[i] for row in rows]
        lengths.append(torch.as_tensor([v.size(0) for v in seqs]))
        features.append(pad_sequence(seqs, batch_first=True))
    indices = torch.stack([row[3].view(-1) for row in rows]).view(len(rows), 1)
    labels = torch.stack([row[4].view(-1) for row in rows]).view(len(rows), 1)
    return features, lengths, indices, labels


def collate_padded(
    rows: Sequence[Sequence[torch.Tensor]],
    max_len: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stand-in for ``_process_2`` after ``Affectdataset`` already padded."""
    stacked = []
    for i in range(3):
        clipped = []
        for row in rows:
            x = row[i][:max_len]
            if x.size(0) < max_len:
                pad = x.new_zeros(max_len - x.size(0), x.size(1))
                x = torch.cat([x, pad], dim=0)
            clipped.append(x)
        stacked.append(torch.stack(clipped, dim=0))
    labels = torch.stack([row[-1].view(-1) for row in rows]).view(len(rows), 1)
    return stacked[0], stacked[1], stacked[2], labels


def zero_ablate(
    streams: Sequence[torch.Tensor],
    keep: Sequence[str],
    names: Sequence[str] = ("visual", "audio", "text"),
) -> List[torch.Tensor]:
    """Replace dropped modalities with zeros, matching the real ablation loader."""
    keep_set = set(keep)
    out = []
    for name, tensor in zip(names, streams):
        out.append(tensor if name in keep_set else torch.zeros_like(tensor))
    return out
