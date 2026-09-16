#!/usr/bin/env python3
"""Show packed-sequence collate versus max-pad collate.

Mirrors ``_process_1`` (Concat / TFN / LMF) and ``_process_2`` (GMTM,
early transformer) in ``model/data/get_dataloader.py``.

    python examples/05_packed_vs_padded.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "model"))

from msa_lab.collate import (  # noqa: E402
    collate_max_pad,
    collate_variable_length,
    packed_nonzero_ratio,
    summarize_collate,
)
from msa_lab.synthetic import make_sentiment_batch  # noqa: E402


def main() -> None:
    batch = make_sentiment_batch(
        batch_size=6,
        seq_len=12,
        preset="toy",
        seed=9,
        variable_lengths=True,
        min_length=4,
    )
    packed = collate_variable_length(batch)
    padded = collate_max_pad(batch, max_pad_num=12)
    rows = summarize_collate(packed, padded)
    ratios = packed_nonzero_ratio(batch)

    print("Variable-length synthetic clips, then two collate styles.\n")
    print(f"{'field':<8}  {'packed (_process_1)':<28}  {'max-pad (_process_2)'}")
    print(f"{'-'*8}  {'-'*28}  {'-'*22}")
    for field, packed_shape, padded_shape in rows:
        print(f"{field:<8}  {packed_shape:<28}  {padded_shape}")
    print()
    print("Fraction of time steps that still carry signal after truncation:")
    for name, ratio in ratios.items():
        print(f"  {name:8s}  {ratio:.3f}")
    print()
    print("Training scripts pick a collate from the fusion method:")
    print("  ConcatEarly / ConcatLate / TFN / LMF / TransformerLate -> packed")
    print("  TransformerEarly / GatedMultiTransformer             -> max-pad 50")


if __name__ == "__main__":
    main()
