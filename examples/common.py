"""Shared paths and synthetic multimodal batches for the CPU walkthroughs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_RESULTS_DIR = MODEL_DIR / "mosi_test"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_DIM = 768
GLOVE_DIM = 300
DEFAULT_SEQ_LEN = 16
DEFAULT_BATCH = 4


def dummy_modalities(
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    text_dim: int = BERT_DIM,
    seed: int = 0,
):
    """Return (visual, audio, text, labels) with MOSEI feature widths.

    Text[:, 0, 0] is forced nonzero so ``Affectdataset`` aligned slicing
    (which searches for the first nonzero text step) would succeed on these
    tensors if they were copied into a pickle dump.
    """
    g = torch.Generator().manual_seed(seed)
    visual = torch.randn(batch, seq_len, VISUAL_DIM, generator=g)
    audio = torch.randn(batch, seq_len, AUDIO_DIM, generator=g)
    text = torch.randn(batch, seq_len, text_dim, generator=g)
    text[:, 0, 0] = 1.0
    labels = torch.linspace(-2.0, 2.0, steps=batch).unsqueeze(1)
    return visual, audio, text, labels


def dummy_packed(batch: int = DEFAULT_BATCH, text_dim: int = BERT_DIM, seed: int = 1):
    """Variable-length streams plus a lengths tensor, as ``_process_1`` would emit."""
    visual, audio, text, labels = dummy_modalities(batch=batch, text_dim=text_dim, seed=seed)
    lengths = torch.tensor([DEFAULT_SEQ_LEN - i for i in range(batch)], dtype=torch.long)
    lengths = torch.clamp(lengths, min=4)
    return [visual, audio, text], [lengths, lengths, lengths], labels
