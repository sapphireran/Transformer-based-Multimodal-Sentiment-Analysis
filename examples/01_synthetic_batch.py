#!/usr/bin/env python3
"""Print MOSI/MOSEI-shaped synthetic batches (BERT and GloVe text widths)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    AUDIO_DIM,
    BERT_DIM,
    GLOVE_DIM,
    VISUAL_DIM,
    describe_batch,
    make_aligned_batch,
)


def main() -> None:
    print("=== BERT-shaped batch (matches mosei_raw_bert.pkl / mosi_raw_bert.pkl) ===")
    vision, audio, text, labels = make_aligned_batch(text_dim=BERT_DIM)
    print(describe_batch(vision, audio, text, labels))
    assert vision.shape[-1] == VISUAL_DIM
    assert audio.shape[-1] == AUDIO_DIM
    assert text.shape[-1] == BERT_DIM

    print()
    print("=== GloVe-shaped batch (matches mosei_raw_glove.pkl) ===")
    vision, audio, text, labels = make_aligned_batch(text_dim=GLOVE_DIM)
    print(describe_batch(vision, audio, text, labels))
    assert text.shape[-1] == GLOVE_DIM

    print()
    print("=== Correlated batch (label depends on mean text; used by mini training) ===")
    vision, audio, text, labels = make_aligned_batch(correlated=True, batch_size=8)
    print(describe_batch(vision, audio, text, labels))
    text_mean = text.mean(dim=(1, 2))
    centered_text = text_mean - text_mean.mean()
    centered_lab = labels.squeeze(1) - labels.mean()
    pearson = (centered_text * centered_lab).sum() / (
        centered_text.norm() * centered_lab.norm() + 1e-8
    )
    print(f"Pearson r(mean text, label) = {pearson.item():+.3f} (should be clearly positive)")
    print()
    print("These tensors are what ConcatEarly would cat to width", VISUAL_DIM + AUDIO_DIM + BERT_DIM)
    print("or, with GloVe text, width", VISUAL_DIM + AUDIO_DIM + GLOVE_DIM)


if __name__ == "__main__":
    main()
