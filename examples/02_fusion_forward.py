#!/usr/bin/env python3
"""Forward every fusion family on one toy batch and print output shapes.

Does not load MOSI / MOSEI pickles or GPU checkpoints.

    python examples/02_fusion_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from msa_lab.fusion_zoo import build_fusion_zoo, describe_fusion_shapes  # noqa: E402
from msa_lab.synthetic import make_sentiment_batch  # noqa: E402


def main() -> None:
    batch = make_sentiment_batch(batch_size=4, seq_len=10, preset="toy", seed=2)
    zoo = build_fusion_zoo(batch.dims())
    rows = describe_fusion_shapes(zoo, batch)

    print("Input: visual/audio/text = "
          f"{tuple(batch.visual.shape)} / {tuple(batch.audio.shape)} / {tuple(batch.text.shape)}")
    print()
    name_w = max(len(row["name"]) for row in rows)
    shape_w = max(len(row["shape"]) for row in rows)
    print(f"{'fusion'.ljust(name_w)}  {'shape'.ljust(shape_w)}  family")
    print(f"{'-' * name_w}  {'-' * shape_w}  ------")
    for row in rows:
        print(f"{row['name'].ljust(name_w)}  {row['shape'].ljust(shape_w)}  {row['family']}")
    print()
    for row in rows:
        print(f"* {row['name']}: {row['notes']}")


if __name__ == "__main__":
    main()
