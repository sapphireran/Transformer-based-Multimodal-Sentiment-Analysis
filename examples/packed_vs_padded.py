#!/usr/bin/env python3
"""Show packed (variable-length) vs padded (fixed-T) collate on toy clips.

Mirrors ``_process_1`` / ``_process_2`` in ``model/data/get_dataloader.py``
without loading a MOSI pickle. Recurrent late fusion uses the packed path;
GMTM and Transformer-early use the stacked cube.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_output_dir  # noqa: E402
from toy_data import AUDIO_DIM, GLOVE_DIM, VISION_DIM, make_toy_batch  # noqa: E402


def make_variable_clips(batch_size: int, max_len: int, seed: int) -> List[Dict[str, np.ndarray]]:
    batch = make_toy_batch(
        batch_size=batch_size, seq_len=max_len, text_backend="glove", seed=seed
    )
    rng = np.random.default_rng(seed + 17)
    lengths = rng.integers(4, max_len + 1, size=batch_size)
    clips = []
    for i, length in enumerate(lengths):
        clips.append(
            {
                "vision": batch.vision[i, :length],
                "audio": batch.audio[i, :length],
                "text": batch.text[i, :length],
                "label": batch.labels[i],
                "length": int(length),
            }
        )
    return clips


def collate_packed(clips: List[Dict[str, np.ndarray]]):
    """``_process_1``-style: pad to the longest clip in the batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    keys = ("vision", "audio", "text")
    stacked = []
    lengths = []
    for key in keys:
        seqs = [torch.from_numpy(clip[key]) for clip in clips]
        lengths.append(torch.tensor([s.size(0) for s in seqs], dtype=torch.long))
        stacked.append(pad_sequence(seqs, batch_first=True))
    labels = torch.tensor(np.stack([clip["label"] for clip in clips]))
    return stacked, lengths, labels


def collate_padded(clips: List[Dict[str, np.ndarray]], max_len: int):
    """``_process_2``-style: right-pad / crop every clip to ``max_len``."""
    import torch
    from torch.nn import functional as F

    cubes = []
    for key, dim in (("vision", VISION_DIM), ("audio", AUDIO_DIM), ("text", GLOVE_DIM)):
        rows = []
        for clip in clips:
            tensor = torch.from_numpy(clip[key][:max_len])
            pad_t = max_len - tensor.shape[0]
            if pad_t > 0:
                tensor = F.pad(tensor, (0, 0, 0, pad_t))
            rows.append(tensor)
        cubes.append(torch.stack(rows))
        assert cubes[-1].shape == (len(clips), max_len, dim)
    labels = torch.tensor(np.stack([clip["label"] for clip in clips]))
    return cubes, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    clips = make_variable_clips(args.batch_size, args.max_len, args.seed)
    raw_lengths = [clip["length"] for clip in clips]
    packed, packed_lengths, packed_labels = collate_packed(clips)
    padded, padded_labels = collate_padded(clips, args.max_len)

    print(f"raw clip lengths: {raw_lengths}")
    print()
    print("packed (_process_1)")
    for name, tensor, lengths in zip(("vision", "audio", "text"), packed, packed_lengths):
        print(f"  {name:<8} {tuple(tensor.shape)}  lengths={lengths.tolist()}")
    print(f"  labels   {tuple(packed_labels.shape)}")
    print()
    print("padded (_process_2, T fixed)")
    for name, tensor in zip(("vision", "audio", "text"), padded):
        print(f"  {name:<8} {tuple(tensor.shape)}")
    print(f"  labels   {tuple(padded_labels.shape)}")
    print()
    print("consumers: packed → Concat/TFN/LMF/TransformerLate (is_packed=True)")
    print("           padded → TransformerEarly / GMTM (is_packed=False)")

    if packed[0].shape[1] != max(raw_lengths):
        print("error: packed T_max should equal the longest raw clip")
        return 1
    if padded[0].shape[1] != args.max_len:
        print("error: padded T should equal --max-len")
        return 1

    if args.write_json:
        payload = {
            "raw_lengths": raw_lengths,
            "packed_shapes": [list(t.shape) for t in packed],
            "padded_shapes": [list(t.shape) for t in padded],
        }
        out = ensure_output_dir() / "packed_vs_padded.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
