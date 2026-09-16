"""Canonical feature widths used by the MOSI / MOSEI loaders.

The training scripts hard-code these numbers in several places
(`train_main_bert.py`, `get_ablation_dataloader`, GMTM `input_dims`).
Keeping one copy here stops the examples from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """One aligned multimodal clip."""

    visual: int = 35
    audio: int = 74
    text: int
    time: int = 50

    @property
    def early_width(self) -> int:
        return self.visual + self.audio + self.text

    @property
    def as_tuple(self) -> tuple[int, int, int]:
        """Order used by every training script: visual, audio, text."""
        return (self.visual, self.audio, self.text)


BERT = FeatureSpec(text=768)
GLOVE = FeatureSpec(text=300)

# Short sequence used by the CPU demos. Real loaders pad to 50.
DEMO_TIME = 16
DEMO_BATCH = 4

MODALITY_INDEX = {"visual": 0, "audio": 1, "text": 2}
MODALITY_ORDER = ("visual", "audio", "text")
