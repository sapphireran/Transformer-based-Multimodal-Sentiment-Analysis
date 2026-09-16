"""Shared paths and MOSI/MOSEI-shaped constants for personal examples."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "output"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# Word-aligned clip length used by every training script.
MAX_SEQ_LEN = 50

# Feature widths from FACET 4.2 / COVAREP / BERT / GloVe.
VISUAL_DIM = 35
AUDIO_DIM = 74
TEXT_BERT_DIM = 768
TEXT_GLOVE_DIM = 300

MODALITY_ORDER = ("visual", "audio", "text")


def device() -> torch.device:
    """Prefer CUDA when the personal box has it; examples must still run on CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
