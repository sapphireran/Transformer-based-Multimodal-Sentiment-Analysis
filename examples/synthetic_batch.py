"""Build aligned multimodal batches with the MOSI/MOSEI ranks.

The real pickles are not in git. This factory is the stand-in used by
the other examples: same axis order, same default widths, labels in
``[-3, 3]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Widths used by train_main_bert.py / train_GMTM_bert.py.
BERT_WIDTHS = {"visual": 35, "audio": 74, "text": 768}
# Widths used by the GloVe scripts.
GLOVE_WIDTHS = {"visual": 35, "audio": 74, "text": 300}


@dataclass(frozen=True)
class SyntheticBatch:
    visual: torch.Tensor  # [B, T, 35]
    audio: torch.Tensor  # [B, T, 74]
    text: torch.Tensor  # [B, T, text_dim]
    labels: torch.Tensor  # [B, 1]
    lengths: torch.Tensor  # [B]  (all T when fully padded)

    def as_list(self) -> list[torch.Tensor]:
        return [self.visual, self.audio, self.text]

    def as_gmtm_input(self) -> list[torch.Tensor]:
        """GMTM expects a list of [B, T, F] tensors (then permutes to T,B,F)."""
        return self.as_list()


def make_synthetic_batch(
    batch_size: int = 4,
    seq_len: int = 12,
    text_dim: int = 32,
    visual_dim: int = 35,
    audio_dim: int = 74,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> SyntheticBatch:
    """Random aligned clip batch.

    ``text_dim`` defaults to 32 (not 768) so CPU demos stay small. Pass
    768 or 300 to mimic a real pickle width.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    visual = torch.randn(batch_size, seq_len, visual_dim, generator=generator)
    audio = torch.randn(batch_size, seq_len, audio_dim, generator=generator)
    text = torch.randn(batch_size, seq_len, text_dim, generator=generator)
    # Sentiment-like targets: squash to [-3, 3].
    raw = torch.randn(batch_size, 1, generator=generator)
    labels = 3.0 * torch.tanh(raw)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    return SyntheticBatch(
        visual=visual.to(device),
        audio=audio.to(device),
        text=text.to(device),
        labels=labels.to(device),
        lengths=lengths.to(device),
    )


def zero_modalities(
    batch: SyntheticBatch, keep: tuple[str, ...]
) -> SyntheticBatch:
    """Reproduce the ablation dataloader: unused streams become zeros."""
    visual = batch.visual if "visual" in keep else torch.zeros_like(batch.visual)
    audio = batch.audio if "audio" in keep else torch.zeros_like(batch.audio)
    text = batch.text if "text" in keep else torch.zeros_like(batch.text)
    return SyntheticBatch(
        visual=visual,
        audio=audio,
        text=text,
        labels=batch.labels,
        lengths=batch.lengths,
    )


def _demo() -> None:
    batch = make_synthetic_batch()
    print("synthetic batch")
    print(f"  visual {tuple(batch.visual.shape)}")
    print(f"  audio  {tuple(batch.audio.shape)}")
    print(f"  text   {tuple(batch.text.shape)}")
    print(f"  labels {tuple(batch.labels.shape)} range "
          f"[{batch.labels.min():.2f}, {batch.labels.max():.2f}]")
    masked = zero_modalities(batch, keep=("text",))
    print(f"  text-only ablation: audio L2={masked.audio.norm():.3f} "
          f"(expect 0.000)")


if __name__ == "__main__":
    _demo()
