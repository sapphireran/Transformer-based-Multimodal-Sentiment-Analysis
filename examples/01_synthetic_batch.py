#!/usr/bin/env python3
"""Print one synthetic MOSI-shaped batch.

Run from the repository root:

    python examples/01_synthetic_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from msa_lab.synthetic import FEATURE_DIMS, make_sentiment_batch  # noqa: E402


def describe(preset: str, seq_len: int = 8, batch_size: int = 4) -> None:
    batch = make_sentiment_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        preset=preset,
        seed=1,
        variable_lengths=True,
    )
    print(f"preset={preset}")
    print(f"  visual {tuple(batch.visual.shape)}  audio {tuple(batch.audio.shape)}  "
          f"text {tuple(batch.text.shape)}")
    print(f"  labels {batch.labels.squeeze(-1).tolist()}")
    print(f"  lengths {batch.lengths.tolist()}")
    print(f"  label range [{float(batch.labels.min()):.2f}, {float(batch.labels.max()):.2f}]")
    print()


def main() -> None:
    print("Synthetic clips use the same [B, T, F] layout as Affectdataset.\n")
    for preset in ("toy", "mosei_bert", "mosei_glove"):
        describe(preset)
    print("Named dimension packs:")
    for name, dims in FEATURE_DIMS.items():
        print(f"  {name:12s}  vis={dims['visual']:3d}  aud={dims['audio']:3d}  txt={dims['text']:3d}")


if __name__ == "__main__":
    main()
