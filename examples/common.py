"""Shared helpers for the CPU MOSI walkthroughs.

The official trainers live under ``model/`` and call ``.cuda()``. Examples stay
on whatever device is available and use MOSI-shaped *synthetic* tensors so they
run without pickles, GloVe, or a GPU.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Real MOSI / MOSEI widths used by the trainers.
MOSI_VISION_DIM = 35
MOSI_AUDIO_DIM = 74
MOSI_BERT_DIM = 768
MOSI_GLOVE_DIM = 300
MOSI_SEQ_LEN = 50

# Tiny widths for GMTM smoke tests (must stay divisible by TinyHParams.num_heads).
TINY_VISION_DIM = 8
TINY_AUDIO_DIM = 10
TINY_TEXT_DIM = 16
TINY_SEQ_LEN = 8


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyHParams:
    """GMTM settings small enough for a laptop CPU step."""

    num_heads = 2
    layers = 1
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 16
    attn_mask = False
    output_dim = 1
    all_steps = False


def _sentiment_from_text(text: np.ndarray) -> np.ndarray:
    """Cheap label: scaled mean of the text channel, clipped to [-3, 3]."""
    raw = text.mean(axis=(1, 2)) * 8.0
    return np.clip(raw, -3.0, 3.0).astype(np.float32)


def make_split(
    n: int,
    seq_len: int,
    vision_dim: int,
    audio_dim: int,
    text_dim: int,
    rng: np.random.Generator,
    prefix: str,
) -> Dict[str, object]:
    vision = rng.normal(0.0, 1.0, size=(n, seq_len, vision_dim)).astype(np.float32)
    audio = rng.normal(0.0, 1.0, size=(n, seq_len, audio_dim)).astype(np.float32)
    text = rng.normal(0.0, 1.0, size=(n, seq_len, text_dim)).astype(np.float32)
    # Guarantee at least one non-zero text frame so Affectdataset alignment works.
    text[:, 0, 0] += 0.5
    labels = _sentiment_from_text(text).reshape(n, 1, 1)
    ids = [f"{prefix}-{i:04d}" for i in range(n)]
    return {
        "vision": vision,
        "audio": audio,
        "text": text,
        "labels": labels,
        "id": ids,
    }


def make_synthetic_mosi(
    n_train: int = 24,
    n_valid: int = 8,
    n_test: int = 8,
    seq_len: int = TINY_SEQ_LEN,
    vision_dim: int = TINY_VISION_DIM,
    audio_dim: int = TINY_AUDIO_DIM,
    text_dim: int = TINY_TEXT_DIM,
    seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Build an in-memory pickle that ``get_dataloader`` can consume."""
    rng = np.random.default_rng(seed)
    return {
        "train": make_split(n_train, seq_len, vision_dim, audio_dim, text_dim, rng, "tr"),
        "valid": make_split(n_valid, seq_len, vision_dim, audio_dim, text_dim, rng, "va"),
        "test": make_split(n_test, seq_len, vision_dim, audio_dim, text_dim, rng, "te"),
    }


def write_synthetic_pickle(path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(make_synthetic_mosi(**kwargs), handle)
    return path


def batch_from_split(
    split: Dict[str, object],
    device_or_str=None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Stack a whole split into one GMTM batch: [vision, audio, text], labels."""
    dev = torch.device(device_or_str) if device_or_str is not None else device()
    vision = torch.from_numpy(np.asarray(split["vision"])).float().to(dev)
    audio = torch.from_numpy(np.asarray(split["audio"])).float().to(dev)
    text = torch.from_numpy(np.asarray(split["text"])).float().to(dev)
    labels = torch.from_numpy(np.asarray(split["labels"])).float().reshape(-1, 1).to(dev)
    return [vision, audio, text], labels


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
