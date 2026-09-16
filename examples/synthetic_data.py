"""Synthetic tensors with the same ranks as the MOSEI / MOSI loaders.

The real pickles are large and gitignored. These helpers let every example
exercise ``model.models`` without downloading CMU data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from examples.common import BERT_LAYOUT, FeatureLayout, seed_everything


@dataclass
class SyntheticBatch:
    vision: torch.Tensor  # [B, T, 35]
    audio: torch.Tensor  # [B, T, 74]
    text: torch.Tensor  # [B, T, 768|300]
    labels: torch.Tensor  # [B, 1]
    layout: FeatureLayout

    def as_list(self) -> List[torch.Tensor]:
        return [self.vision, self.audio, self.text]

    def as_padded_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Matches ``_process_2``: vision, audio, text, labels."""
        return self.vision, self.audio, self.text, self.labels


def _text_driven_labels(text: torch.Tensor) -> torch.Tensor:
    """Map mean text activation into (-3, 3) so a toy model has a signal."""
    scores = text.mean(dim=(1, 2))
    scores = 3.0 * torch.tanh(scores)
    return scores.unsqueeze(1)


def make_batch(
    batch_size: int = 4,
    layout: FeatureLayout | None = None,
    seq_len: int | None = None,
    seed: int = 2024,
    correlated_labels: bool = True,
) -> SyntheticBatch:
    layout = layout or BERT_LAYOUT
    seq_len = seq_len if seq_len is not None else layout.seq_len
    seed_everything(seed)
    vision = torch.randn(batch_size, seq_len, layout.visual)
    audio = torch.randn(batch_size, seq_len, layout.audio)
    text = torch.randn(batch_size, seq_len, layout.text)
    if correlated_labels:
        labels = _text_driven_labels(text)
    else:
        labels = (torch.rand(batch_size, 1) * 6.0) - 3.0
    return SyntheticBatch(vision, audio, text, labels, layout)


def make_unimodal_batch(
    keep: Sequence[str],
    batch: SyntheticBatch | None = None,
) -> SyntheticBatch:
    """Zero the modalities that are not listed — same contract as the loaders."""
    batch = batch or make_batch()
    keep_set = {name.lower() for name in keep}
    vision = batch.vision if "visual" in keep_set or "vision" in keep_set else torch.zeros_like(batch.vision)
    audio = batch.audio if "audio" in keep_set else torch.zeros_like(batch.audio)
    text = batch.text if "text" in keep_set else torch.zeros_like(batch.text)
    return SyntheticBatch(vision, audio, text, batch.labels, batch.layout)


def make_pickle_dict(
    n_train: int = 8,
    n_valid: int = 4,
    n_test: int = 4,
    layout: FeatureLayout | None = None,
    seed: int = 7,
) -> Dict[str, dict]:
    """In-memory stand-in for ``mosei_raw_bert.pkl`` / ``mosi_raw_*.pkl``."""
    layout = layout or BERT_LAYOUT
    seed_everything(seed)
    out: Dict[str, dict] = {}
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        vision = torch.randn(n, layout.seq_len, layout.visual).numpy()
        audio = torch.randn(n, layout.seq_len, layout.audio).numpy()
        text = torch.randn(n, layout.seq_len, layout.text).numpy()
        labels = (torch.rand(n, 1, 1) * 6.0 - 3.0).numpy()
        ids = [f"synthetic_{split}_{i}" for i in range(n)]
        out[split] = {
            "vision": vision,
            "audio": audio,
            "text": text,
            "labels": labels,
            "id": ids,
        }
    return out


def describe_batch(batch: SyntheticBatch) -> list[tuple[str, str]]:
    return [
        ("vision", f"{tuple(batch.vision.shape)} dtype={batch.vision.dtype}"),
        ("audio", f"{tuple(batch.audio.shape)} dtype={batch.audio.dtype}"),
        ("text", f"{tuple(batch.text.shape)} dtype={batch.text.dtype}"),
        ("labels", f"{tuple(batch.labels.shape)} range=[{batch.labels.min():.2f}, {batch.labels.max():.2f}]"),
        ("layout", f"{batch.layout.name} F={batch.layout.as_list} T={batch.vision.size(1)}"),
    ]


def main() -> int:
    from examples.common import GLOVE_LAYOUT, print_kv

    print("Synthetic BERT batch (default max_pad ranks)")
    print_kv(describe_batch(make_batch(batch_size=4, layout=BERT_LAYOUT)))
    print("Synthetic GloVe batch")
    print_kv(describe_batch(make_batch(batch_size=2, layout=GLOVE_LAYOUT, seed=1)))
    blob = make_pickle_dict()
    print("Pickle-shaped dict splits:", {k: blob[k]["vision"].shape[0] for k in blob})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
