"""Build tiny MOSI-shaped multimodal batches without CMU pickles.

The real loaders read

    {train,valid,test} -> {vision, audio, text, labels, id}

from a pickle. This module speaks the same language so examples and tests
can exercise fusion / GMTM / a toy trainer on a laptop.

Labels live in [-3, 3]. When ``text_correlated=True`` the score is a noisy
function of the text-channel mean so a small MLP/GMTM has something to fit.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    AUDIO_DIM,
    MAX_SEQ_LEN,
    TEXT_BERT_DIM,
    TEXT_GLOVE_DIM,
    VISUAL_DIM,
)

SPLIT_NAMES = ("train", "valid", "test")


@dataclass(frozen=True)
class SyntheticConfig:
    n_train: int = 64
    n_valid: int = 16
    n_test: int = 16
    seq_len: int = MAX_SEQ_LEN
    visual_dim: int = VISUAL_DIM
    audio_dim: int = AUDIO_DIM
    text_dim: int = TEXT_BERT_DIM
    text_correlated: bool = True
    label_noise: float = 0.35
    seed: int = 7


def _split_size(cfg: SyntheticConfig, split: str) -> int:
    return {"train": cfg.n_train, "valid": cfg.n_valid, "test": cfg.n_test}[split]


def _make_split(cfg: SyntheticConfig, split: str, rng: np.random.Generator) -> dict:
    n = _split_size(cfg, split)
    t, dv, da, dt = cfg.seq_len, cfg.visual_dim, cfg.audio_dim, cfg.text_dim

    vision = rng.normal(0.0, 1.0, size=(n, t, dv)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(n, t, da)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(n, t, dt)).astype(np.float32)

    # Drop a couple of leading frames to 0 so Affectdataset-style
    # "first nonzero text row" trimming has something to do if reused.
    if t >= 4:
        text[:, :2, :] = 0.0

    if cfg.text_correlated:
        # Signed magnitude of the text stream, squashed into [-3, 3].
        strength = text[:, 2:, :].mean(axis=(1, 2))
        strength = np.tanh(strength * 1.8) * 3.0
        noise = rng.normal(0.0, cfg.label_noise, size=(n,))
        labels = np.clip(strength + noise, -3.0, 3.0).astype(np.float32)
    else:
        labels = rng.uniform(-3.0, 3.0, size=(n,)).astype(np.float32)

    # Match the wide MOSEI label layout: first column is sentiment.
    labels_wide = np.zeros((n, 1, 8), dtype=np.float32)
    labels_wide[:, 0, 0] = labels

    ids = [f"synthetic_{split}_{i}" for i in range(n)]
    return {
        "vision": vision,
        "audio": audio,
        "text": text,
        "labels": labels_wide,
        "id": ids,
    }


def build_synthetic_dict(cfg: Optional[SyntheticConfig] = None) -> dict:
    """Return a pickle-compatible MOSI/MOSEI dict."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    return {split: _make_split(cfg, split, rng) for split in SPLIT_NAMES}


def save_synthetic_pickle(path, cfg: Optional[SyntheticConfig] = None) -> dict:
    data = build_synthetic_dict(cfg)
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


class SyntheticAffectDataset(Dataset):
    """Minimal ``[vision, audio, text, y]`` dataset used by the examples.

    This is *not* a copy of ``Affectdataset``. It skips alignment trimming
    and always returns max-padded clips so fusion demos stay shape-stable.
    """

    def __init__(self, split_dict: dict, modalities: Optional[Sequence[str]] = None):
        self.vision = torch.from_numpy(np.asarray(split_dict["vision"]))
        self.audio = torch.from_numpy(np.asarray(split_dict["audio"]))
        self.text = torch.from_numpy(np.asarray(split_dict["text"]))
        labels = np.asarray(split_dict["labels"])
        if labels.ndim == 3:
            self.labels = torch.from_numpy(labels[:, 0, 0]).float().unsqueeze(1)
        elif labels.ndim == 2:
            self.labels = torch.from_numpy(labels[:, 0]).float().unsqueeze(1)
        else:
            self.labels = torch.from_numpy(labels).float().view(-1, 1)
        self.modalities = tuple(modalities) if modalities else ("visual", "audio", "text")

    def __len__(self) -> int:
        return int(self.vision.shape[0])

    def _maybe_zero(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name in self.modalities:
            return tensor
        return torch.zeros_like(tensor)

    def __getitem__(self, idx: int):
        vision = self._maybe_zero("visual", self.vision[idx])
        audio = self._maybe_zero("audio", self.audio[idx])
        text = self._maybe_zero("text", self.text[idx])
        return vision.float(), audio.float(), text.float(), self.labels[idx]


def collate_maxpad(batch: List[Tuple[torch.Tensor, ...]]):
    vision, audio, text, labels = zip(*batch)
    return (
        torch.stack(vision, 0),
        torch.stack(audio, 0),
        torch.stack(text, 0),
        torch.stack(labels, 0),
    )


def make_dataloaders(
    cfg: Optional[SyntheticConfig] = None,
    batch_size: int = 8,
    modalities: Optional[Sequence[str]] = None,
) -> Dict[str, DataLoader]:
    data = build_synthetic_dict(cfg)
    loaders = {}
    for split, payload in data.items():
        ds = SyntheticAffectDataset(payload, modalities=modalities)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_maxpad,
        )
    return loaders


def random_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    text_dim: int = TEXT_BERT_DIM,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uncorrelated Gaussian batch for shape / smoke tests."""
    g = torch.Generator()
    g.manual_seed(0)
    vision = torch.randn(batch_size, seq_len, VISUAL_DIM, generator=g)
    audio = torch.randn(batch_size, seq_len, AUDIO_DIM, generator=g)
    text = torch.randn(batch_size, seq_len, text_dim, generator=g)
    if device is not None:
        vision, audio, text = vision.to(device), audio.to(device), text.to(device)
    return vision, audio, text


def zero_mask(
    vision: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    keep: Iterable[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ``get_ablation_dataloader`` zero-masking on a live batch."""
    keep = set(keep)
    if "visual" not in keep:
        vision = torch.zeros_like(vision)
    if "audio" not in keep:
        audio = torch.zeros_like(audio)
    if "text" not in keep:
        text = torch.zeros_like(text)
    return vision, audio, text


def glove_config(**kwargs) -> SyntheticConfig:
    kwargs.setdefault("text_dim", TEXT_GLOVE_DIM)
    return SyntheticConfig(**kwargs)


if __name__ == "__main__":
    cfg = SyntheticConfig(n_train=8, n_valid=4, n_test=4, text_dim=32, seq_len=12)
    data = build_synthetic_dict(cfg)
    for split, payload in data.items():
        print(
            f"{split:5s}  vision{tuple(payload['vision'].shape)}  "
            f"audio{tuple(payload['audio'].shape)}  "
            f"text{tuple(payload['text'].shape)}  "
            f"y∈[{payload['labels'][:, 0, 0].min():.2f}, "
            f"{payload['labels'][:, 0, 0].max():.2f}]"
        )
