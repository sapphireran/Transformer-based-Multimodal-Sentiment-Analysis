"""Shared helpers for the personal MOSI/MOSEI-shaped examples.

The real training scripts read large pickles. These helpers only allocate
tensors with the same widths the loaders would have produced:

    vision [B, T, 35]   Facet 4.2
    audio  [B, T, 74]   COVAREP
    text   [B, T, 768]  BERT   or  [B, T, 300] GloVe
    label  [B, 1]       sentiment in [-3, 3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

VISUAL_DIM = 35
AUDIO_DIM = 74
BERT_DIM = 768
GLOVE_DIM = 300
DEFAULT_SEQ_LEN = 16
DEFAULT_BATCH = 4


def device_of() -> torch.device:
    """Examples stay on CPU so they run in this cloud workspace."""
    return torch.device("cpu")


def make_aligned_batch(
    batch_size: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    text_dim: int = BERT_DIM,
    correlated: bool = False,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one aligned (vision, audio, text, label) batch.

    When ``correlated`` is true the label is a noisy affine function of the
    mean text channel. That gives ``05_mini_training.py`` a learnable
    signal without shipping any CMU clip.
    """
    device = device or device_of()
    kwargs = {"device": device}
    if generator is not None:
        kwargs["generator"] = generator

    vision = torch.randn(batch_size, seq_len, VISUAL_DIM, **kwargs)
    audio = torch.randn(batch_size, seq_len, AUDIO_DIM, **kwargs)
    text = torch.randn(batch_size, seq_len, text_dim, **kwargs)

    if correlated:
        # Plant a clip-level scalar in every text frame, then use a monotone
        # map of that scalar as the label. Mean-pooling raw Gaussian text is
        # ~N(0, 1/sqrt(T·F)) and would be drowned by even small label noise.
        latent = torch.randn(batch_size, 1, **kwargs)
        text = text + latent.unsqueeze(1)
        labels = torch.tanh(latent) * 3.0
    else:
        labels = torch.rand(batch_size, 1, **kwargs) * 6.0 - 3.0

    return vision, audio, text, labels


def packed_lengths(batch_size: int, seq_len: int, device: torch.device | None = None) -> list[torch.Tensor]:
    """Dummy lengths (all ``seq_len``) matching ``_process_1``'s three-stream list."""
    device = device or device_of()
    full = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    return [full.clone(), full.clone(), full.clone()]


def describe_batch(
    vision: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    labels: torch.Tensor,
) -> str:
    lines = [
        f"vision {tuple(vision.shape)} dtype={vision.dtype}",
        f"audio  {tuple(audio.shape)} dtype={audio.dtype}",
        f"text   {tuple(text.shape)} dtype={text.dtype}",
        f"label  {tuple(labels.shape)} "
        f"range=[{labels.min().item():.3f}, {labels.max().item():.3f}]",
    ]
    return "\n".join(lines)
