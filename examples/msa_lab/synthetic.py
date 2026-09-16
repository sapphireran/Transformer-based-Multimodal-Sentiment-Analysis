"""Synthetic aligned multimodal clips that mimic MOSI / MOSEI tensors.

The real CMU pickles are not in this repository. The examples therefore
build small, fully observed batches with the same axis convention the
training scripts expect:

* visual / audio / text are ``[batch, time, feature]`` float tensors
* labels are ``[batch, 1]`` continuous scores in ``[-3, 3]``
* a shared latent sentiment drives every modality, plus per-modality noise

That correlation is enough for a tiny gated transformer to reduce MAE in
a few CPU steps without downloading Facet / COVAREP / BERT features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

FEATURE_DIMS: dict[str, dict[str, int]] = {
    "mosei_bert": {"visual": 35, "audio": 74, "text": 768},
    "mosei_glove": {"visual": 35, "audio": 74, "text": 300},
    "mosi_bert": {"visual": 35, "audio": 74, "text": 768},
    "mosi_glove": {"visual": 35, "audio": 74, "text": 300},
    "toy": {"visual": 8, "audio": 12, "text": 16},
}

MODALITY_ORDER: tuple[str, ...] = ("visual", "audio", "text")


@dataclass
class SentimentBatch:
    """One packed-or-padded toy batch plus the latent score that generated it."""

    visual: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    preset: str

    def as_list(self) -> list[torch.Tensor]:
        """``[vision, audio, text]`` in the order ``Affectdataset`` uses."""
        return [self.visual, self.audio, self.text]

    def dims(self) -> dict[str, int]:
        return {
            "visual": int(self.visual.size(-1)),
            "audio": int(self.audio.size(-1)),
            "text": int(self.text.size(-1)),
        }

    def to(self, device: torch.device | str) -> "SentimentBatch":
        return SentimentBatch(
            visual=self.visual.to(device),
            audio=self.audio.to(device),
            text=self.text.to(device),
            labels=self.labels.to(device),
            lengths=self.lengths.to(device),
            preset=self.preset,
        )


def _resolve_dims(preset: str, dims: Mapping[str, int] | None) -> dict[str, int]:
    if dims is not None:
        missing = [name for name in MODALITY_ORDER if name not in dims]
        if missing:
            raise KeyError(f"dims missing modalities {missing}")
        return {name: int(dims[name]) for name in MODALITY_ORDER}
    if preset not in FEATURE_DIMS:
        known = ", ".join(sorted(FEATURE_DIMS))
        raise KeyError(f"unknown preset {preset!r}; choose one of {known}")
    return dict(FEATURE_DIMS[preset])


def _modality_tensor(
    labels: torch.Tensor,
    seq_len: int,
    feature_dim: int,
    phase: float,
    noise: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Expand a scalar sentiment into a time-varying feature stream.

    The latent score in ``[-3, 3]`` is scaled to ``[-1, 1]`` and broadcast
    through a sinusoidal template plus a learned-looking random projection.
    Noise is isotropic Gaussian. The result is *not* a substitute for
    Facet42 / COVAREP / BERT; it only preserves the tensor shapes and a
    learnable correlation with the label.
    """
    batch = labels.size(0)
    device = labels.device
    latent = (labels / 3.0).to(device)  # [B, 1]
    time = torch.linspace(0.0, 1.0, seq_len, device=device).view(1, seq_len, 1)
    envelope = latent.view(batch, 1, 1) * torch.sin(2.0 * torch.pi * (time + phase))
    projection = torch.randn(
        1, 1, feature_dim, generator=generator, device=device
    )
    signal = envelope * projection
    scale = torch.linspace(0.4, 1.0, feature_dim, device=device).view(1, 1, feature_dim)
    signal = signal * scale
    gauss = torch.randn(
        batch, seq_len, feature_dim, generator=generator, device=device
    )
    return (signal + noise * gauss).float()


def make_sentiment_batch(
    batch_size: int = 8,
    seq_len: int = 16,
    preset: str = "toy",
    dims: Mapping[str, int] | None = None,
    noise: float = 0.25,
    seed: int = 0,
    device: torch.device | str = "cpu",
    variable_lengths: bool = False,
    min_length: int = 4,
) -> SentimentBatch:
    """Build one aligned multimodal batch.

    Parameters
    ----------
    batch_size, seq_len:
        Clip count and (maximum) time steps. Real training uses 32 / 50.
    preset:
        Named dimension pack (``toy``, ``mosei_bert``, ``mosi_glove``, ...).
    dims:
        Optional explicit ``{visual, audio, text}`` sizes; overrides preset.
    noise:
        Gaussian stddev. Raise it to make the toy regression harder.
    variable_lengths:
        If true, each clip is truncated then zero-padded so collate demos
        can show packed-sequence lengths.
    """
    if batch_size < 1 or seq_len < 2:
        raise ValueError("batch_size >= 1 and seq_len >= 2 are required")
    device_t = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    resolved = _resolve_dims(preset, dims)

    labels = (torch.rand(batch_size, 1, generator=generator) * 6.0) - 3.0
    # Keep a few strictly positive / negative / near-zero scores so binary
    # metrics in the walkthrough have something to split on.
    if batch_size >= 4:
        labels[0, 0] = 2.4
        labels[1, 0] = -1.8
        labels[2, 0] = 0.0
        labels[3, 0] = 0.7

    visual = _modality_tensor(labels, seq_len, resolved["visual"], 0.00, noise, generator)
    audio = _modality_tensor(labels, seq_len, resolved["audio"], 0.17, noise, generator)
    text = _modality_tensor(labels, seq_len, resolved["text"], 0.31, noise, generator)

    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    if variable_lengths:
        span = max(seq_len - min_length, 1)
        extra = torch.randint(
            0, span + 1, (batch_size,), generator=generator
        )
        lengths = torch.clamp(min_length + extra, max=seq_len)
        for index, length in enumerate(lengths.tolist()):
            visual[index, length:] = 0
            audio[index, length:] = 0
            text[index, length:] = 0

    return SentimentBatch(
        visual=visual.to(device_t),
        audio=audio.to(device_t),
        text=text.to(device_t),
        labels=labels.to(device_t),
        lengths=lengths.to(device_t),
        preset=preset,
    )


def zero_modalities(
    batch: SentimentBatch,
    drop: Iterable[str],
) -> SentimentBatch:
    """Ablation helper: replace dropped modalities with zeros.

    ``get_ablation_dataloader`` does the same thing on real MOSI / MOSEI
    rows so the gated model can keep a fixed three-encoder signature.
    """
    drop_set = {name.lower() for name in drop}
    unknown = drop_set - set(MODALITY_ORDER)
    if unknown:
        raise KeyError(f"cannot drop unknown modalities {sorted(unknown)}")
    visual = torch.zeros_like(batch.visual) if "visual" in drop_set else batch.visual.clone()
    audio = torch.zeros_like(batch.audio) if "audio" in drop_set else batch.audio.clone()
    text = torch.zeros_like(batch.text) if "text" in drop_set else batch.text.clone()
    return SentimentBatch(
        visual=visual,
        audio=audio,
        text=text,
        labels=batch.labels.clone(),
        lengths=batch.lengths.clone(),
        preset=batch.preset,
    )


def iter_presets() -> Sequence[str]:
    return tuple(FEATURE_DIMS.keys())
