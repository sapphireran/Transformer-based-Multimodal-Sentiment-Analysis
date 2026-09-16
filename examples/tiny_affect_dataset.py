#!/usr/bin/env python3
"""Build a MOSI/MOSEI-shaped pickle dict and a tiny Dataset over it.

This does not import model/data/get_dataloader.py (that module imports
torchtext even for the pickle path). The class below follows the same
sample contract: [vision, audio, text, label].
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.synthetic import feature_widths, make_tiny_affect_dict


class TinyAffectDataset:
    """Minimal stand-in for model.data.get_dataloader.Affectdataset."""

    def __init__(self, split: Dict[str, object], max_pad_num: int = 50):
        self.vision = np.asarray(split["vision"])
        self.audio = np.asarray(split["audio"])
        self.text = np.asarray(split["text"])
        self.labels = np.asarray(split["labels"])
        self.ids = list(split["id"])
        self.max_pad_num = max_pad_num
        if self.text.sum() == 0:
            raise ValueError("text split is entirely zeros; drop_entry would remove it")

    def __len__(self) -> int:
        return int(self.vision.shape[0])

    def __getitem__(self, index: int) -> List[object]:
        vision = self.vision[index, : self.max_pad_num]
        audio = self.audio[index, : self.max_pad_num]
        text = self.text[index, : self.max_pad_num]
        # Alignment start: first nonzero text row, same idea as Affectdataset.
        nonzero = np.flatnonzero(np.abs(text).sum(axis=1))
        start = int(nonzero[0]) if nonzero.size else 0
        label = np.asarray(self.labels[index, 0, 0], dtype=np.float32).reshape(1)
        return [vision[start:], audio[start:], text[start:], label]

    def describe(self) -> str:
        return (
            f"n={len(self)}  vision={self.vision.shape}  audio={self.audio.shape}  "
            f"text={self.text.shape}  labels={self.labels.shape}"
        )


def collate_max_pad(samples: List[List[object]], max_pad_num: int = 50):
    """Stack variable-length samples into [B, T, F] like _process_2."""
    import torch

    def _pad(seq: np.ndarray, width: int) -> np.ndarray:
        clipped = seq[:max_pad_num]
        out = np.zeros((max_pad_num, width), dtype=np.float32)
        out[: clipped.shape[0]] = clipped
        return out

    vision = np.stack([_pad(s[0], s[0].shape[1]) for s in samples])
    audio = np.stack([_pad(s[1], s[1].shape[1]) for s in samples])
    text = np.stack([_pad(s[2], s[2].shape[1]) for s in samples])
    labels = np.stack([s[3] for s in samples])
    return (
        torch.tensor(vision),
        torch.tensor(audio),
        torch.tensor(text),
        torch.tensor(labels),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=("bert", "glove"), default="bert")
    parser.add_argument("--write-pkl", type=Path, default=None)
    args = parser.parse_args(argv)

    blob = make_tiny_affect_dict(embedding=args.embedding)
    widths = feature_widths(args.embedding)
    print(f"embedding={args.embedding}  widths={widths}")
    for split_name, split in blob.items():
        dataset = TinyAffectDataset(split)
        print(f"  {split_name:5s}  {dataset.describe()}  first_id={dataset.ids[0]}")
        sample = dataset[0]
        print(
            f"         sample0 ranks: vision{sample[0].shape} audio{sample[1].shape} "
            f"text{sample[2].shape} label{sample[3].shape}={float(sample[3][0]):.3f}"
        )

    try:
        import torch  # noqa: F401

        batch = collate_max_pad([TinyAffectDataset(blob["train"])[i] for i in range(3)])
        print(
            "collate_max_pad train[:3] -> "
            + ", ".join(f"{name}{tuple(t.shape)}" for name, t in zip(
                ("vision", "audio", "text", "label"), batch
            ))
        )
    except ImportError:
        print("torch not installed; skipped collate_max_pad demo")

    if args.write_pkl is not None:
        args.write_pkl.parent.mkdir(parents=True, exist_ok=True)
        with args.write_pkl.open("wb") as handle:
            pickle.dump(blob, handle)
        print(f"wrote {args.write_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
