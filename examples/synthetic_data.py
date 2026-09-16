"""Build MOSI/MOSEI-shaped tensors without downloading the corpora.

The real loaders read a pickle of aligned numpy arrays and yield either
a packed batch (`_process_1`) or a max-padded cube (`_process_2`). The
helpers here only reconstruct the *tensor shapes* those collate
functions produce, plus a label in ``[-3, 3]``. That is enough to
exercise every fusion module and GMTM on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from examples.shapes import BERT, DEMO_BATCH, DEMO_TIME, FeatureSpec, MODALITY_INDEX, MODALITY_ORDER


@dataclass
class SyntheticBatch:
    """One aligned batch.

    Attributes
    ----------
    vision, audio, text:
        ``[B, T, F]`` float tensors, the max-pad layout GMTM and
        TransformerEarly consume.
    labels:
        ``[B, 1]`` sentiment scores in ``[-3, 3]``.
    spec:
        Feature widths used to draw the batch.
    """

    vision: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor
    spec: FeatureSpec

    def as_list(self) -> list[torch.Tensor]:
        return [self.vision, self.audio, self.text]

    def widths(self) -> tuple[int, int, int]:
        return self.spec.as_tuple

    def zero_modalities(self, drop: Iterable[str]) -> "SyntheticBatch":
        """Return a copy with the named streams replaced by zeros.

        Matches ``get_ablation_dataloader``: dropped modalities stay in
        the batch so GMTM always sees three inputs.
        """
        streams = {
            "visual": self.vision.clone(),
            "audio": self.audio.clone(),
            "text": self.text.clone(),
        }
        for name in drop:
            if name not in streams:
                raise KeyError(f"unknown modality {name!r}; expected one of {MODALITY_ORDER}")
            streams[name].zero_()
        return SyntheticBatch(
            vision=streams["visual"],
            audio=streams["audio"],
            text=streams["text"],
            labels=self.labels.clone(),
            spec=self.spec,
        )


def _draw_features(batch: int, time: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(batch, time, dim, generator=generator)


def _draw_labels(batch: int, generator: torch.Generator) -> torch.Tensor:
    # MOSI/MOSEI scores are bounded. A tanh keeps the synthetic labels
    # in the same interval the Acc-7 / Acc-5 binning assumes.
    raw = torch.randn(batch, 1, generator=generator)
    return 3.0 * torch.tanh(raw)


def make_batch(
    spec: FeatureSpec = BERT,
    batch_size: int = DEMO_BATCH,
    seq_len: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    """Draw one max-padded batch.

    Parameters
    ----------
    spec:
        BERT (text=768) or GLOVE (text=300), or any custom ``FeatureSpec``.
    batch_size, seq_len:
        Defaults are small so the demos finish on CPU in seconds.
    seed:
        Passed to a local ``torch.Generator`` so two demos with the
        same seed draw the same batch.
    """
    time = spec.time if seq_len is None else seq_len
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vision = _draw_features(batch_size, time, spec.visual, generator)
    audio = _draw_features(batch_size, time, spec.audio, generator)
    text = _draw_features(batch_size, time, spec.text, generator)
    labels = _draw_labels(batch_size, generator)
    device = torch.device(device)
    return SyntheticBatch(
        vision=vision.to(device),
        audio=audio.to(device),
        text=text.to(device),
        labels=labels.to(device),
        spec=spec,
    )


def make_vector_batch(
    dims: Sequence[int],
    batch_size: int = DEMO_BATCH,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Draw already-pooled per-modality vectors.

    ConcatLate, TensorFusion, LowRankTensorFusion, and TransformerFusion
    all consume ``[B, D_i]`` (or ``[B, D_i]``-like) lists. Using a
    dedicated helper keeps the fusion demo from pretending those
    methods still see a time axis.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = torch.device(device)
    return [torch.randn(batch_size, d, generator=generator).to(device) for d in dims]


def describe_batch(batch: SyntheticBatch) -> str:
    lines = [
        f"spec: visual={batch.spec.visual} audio={batch.spec.audio} "
        f"text={batch.spec.text} time={batch.vision.shape[1]}",
        f"vision {tuple(batch.vision.shape)}  audio {tuple(batch.audio.shape)}  "
        f"text {tuple(batch.text.shape)}  labels {tuple(batch.labels.shape)}",
        f"label range [{batch.labels.min().item():+.3f}, {batch.labels.max().item():+.3f}]",
    ]
    return "\n".join(lines)


# Silence unused-import lint on the public re-export.
_ = (MODALITY_INDEX,)
