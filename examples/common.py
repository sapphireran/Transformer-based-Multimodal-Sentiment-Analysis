"""Shared paths, feature widths, and device helpers for the examples."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# Widths used by train_main_bert.py / train_main_glove.py.
VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_TEXT_DIM = 768
GLOVE_TEXT_DIM = 300
MAX_SEQ_LEN = 50

BERT_EARLY_DIM = VISUAL_DIM + AUDIO_DIM + BERT_TEXT_DIM  # 877
GLOVE_EARLY_DIM = VISUAL_DIM + AUDIO_DIM + GLOVE_TEXT_DIM  # 409


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def feature_dims(text: str = "bert") -> tuple[int, int, int]:
    if text == "bert":
        return VISUAL_DIM, AUDIO_DIM, BERT_TEXT_DIM
    if text == "glove":
        return VISUAL_DIM, AUDIO_DIM, GLOVE_TEXT_DIM
    raise ValueError(f"text must be 'bert' or 'glove', got {text!r}")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
