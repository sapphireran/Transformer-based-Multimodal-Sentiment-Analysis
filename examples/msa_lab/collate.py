"""Packed vs max-padded collate, mirroring ``get_dataloader._process_1/2``.

``_process_1`` (used by Concat / TFN / LMF / late transformers) keeps the
true sequence length next to a padded tensor so LSTM/GRU can call
``pack_padded_sequence``.

``_process_2`` (used by early transformers and GMTM) slices every clip to
``max_pad_num`` (50 in the training scripts) and zero-pads the rest so a
Transformer sees a dense ``[B, T, F]`` batch.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .synthetic import SentimentBatch


def collate_variable_length(batch: SentimentBatch) -> dict[str, torch.Tensor | list]:
    """Rebuild a packed-style batch from possibly truncated clips."""
    visuals, audios, texts = [], [], []
    for index, length in enumerate(batch.lengths.tolist()):
        visuals.append(batch.visual[index, :length])
        audios.append(batch.audio[index, :length])
        texts.append(batch.text[index, :length])

    padded_visual = pad_sequence(visuals, batch_first=True)
    padded_audio = pad_sequence(audios, batch_first=True)
    padded_text = pad_sequence(texts, batch_first=True)
    return {
        "vision": padded_visual,
        "audio": padded_audio,
        "text": padded_text,
        "lengths": batch.lengths.clone(),
        "labels": batch.labels.clone(),
        "style": "process_1_packed",
    }


def collate_max_pad(batch: SentimentBatch, max_pad_num: int | None = None) -> dict[str, torch.Tensor | str]:
    """Dense ``[B, T, F]`` batch used by GMTM / EarlyFusionTransformer."""
    if max_pad_num is None:
        max_pad_num = int(batch.visual.size(1))
    def pad_mod(tensor: torch.Tensor) -> torch.Tensor:
        clipped = tensor[:, :max_pad_num]
        pad_t = max_pad_num - clipped.size(1)
        if pad_t > 0:
            clipped = torch.nn.functional.pad(clipped, (0, 0, 0, pad_t))
        return clipped

    return {
        "vision": pad_mod(batch.visual),
        "audio": pad_mod(batch.audio),
        "text": pad_mod(batch.text),
        "labels": batch.labels.clone(),
        "style": "process_2_max_pad",
        "max_pad_num": torch.tensor(max_pad_num),
    }


def summarize_collate(packed: dict, padded: dict) -> list[tuple[str, str, str]]:
    """Human-readable shape table for the packed-vs-padded example."""
    rows = []
    for key in ("vision", "audio", "text"):
        rows.append(
            (
                key,
                "x".join(str(int(dim)) for dim in packed[key].shape),
                "x".join(str(int(dim)) for dim in padded[key].shape),
            )
        )
    rows.append(
        (
            "lengths",
            ",".join(str(int(value)) for value in packed["lengths"].tolist()),
            "dense (no lengths tensor)",
        )
    )
    return rows


def packed_nonzero_ratio(batch: SentimentBatch) -> dict[str, float]:
    """Fraction of non-zero time steps; useful when lengths vary."""
    ratios = {}
    for name, tensor in (
        ("visual", batch.visual),
        ("audio", batch.audio),
        ("text", batch.text),
    ):
        time_active = (tensor.abs().sum(dim=-1) > 0).float().mean().item()
        ratios[name] = float(time_active)
    return ratios
