#!/usr/bin/env python3
"""Build an in-memory MOSI-shaped dict and run the real dataset + collate path.

This is what ``get_dataloader`` does after ``pickle.load``, minus the file.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import AUDIO_DIM, BERT_DIM, VISUAL_DIM
from data.get_dataloader import Affectdataset, _process_1, _process_2


def make_split(n: int = 8, seq_len: int = 20, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    vision = rng.normal(size=(n, seq_len, VISUAL_DIM)).astype(np.float32)
    audio = rng.normal(size=(n, seq_len, AUDIO_DIM)).astype(np.float32)
    text = rng.normal(size=(n, seq_len, BERT_DIM)).astype(np.float32)
    # Leading audio frames look like COVAREP unvoiced bins; the dataset zeros -inf.
    audio[0, 0, 0] = -np.inf
    # Guarantee a nonzero text step so aligned=True can find a start index.
    text[:, 0, 0] = 1.0
    # Two clips start later — aligned slicing should shorten them.
    text[1, :3, :] = 0.0
    text[1, 3, 0] = 1.0
    labels = rng.uniform(-2.5, 2.5, size=(n, 1, 1)).astype(np.float32)
    # Multi-column label row: collate should keep the first score only.
    extra = rng.random((n, 1, 3)).astype(np.float32)
    extra[:, :, 0] = labels[:, :, 0]
    return {
        "vision": vision,
        "audio": audio,
        "text": text,
        "labels": extra,
        "id": [f"synthetic[{i}]" for i in range(n)],
    }


def _show_batch(name: str, batch) -> None:
    print(f"\n{name}")
    if name.startswith("packed"):
        streams, lengths, indices, labels = batch
        print(f"  streams:  {[tuple(s.shape) for s in streams]}")
        print(f"  lengths:  {[t.tolist() for t in lengths]}")
        print(f"  indices:  {tuple(indices.shape)}  {indices.view(-1).tolist()}")
        print(f"  labels:   {tuple(labels.shape)}  {labels.view(-1).tolist()}")
    else:
        visual, audio, text, labels = batch
        print(f"  visual {tuple(visual.shape)}  audio {tuple(audio.shape)}  text {tuple(text.shape)}")
        print(f"  labels {tuple(labels.shape)}  {labels.view(-1).tolist()}")


def main() -> None:
    split = make_split()
    print("In-memory split")
    for key, value in split.items():
        if key == "id":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {np.shape(value)} dtype={getattr(value, 'dtype', type(value))}")

    packed_ds = Affectdataset(split, flatten_time_series=False, aligned=True, max_pad=False)
    padded_ds = Affectdataset(split, flatten_time_series=False, aligned=True, max_pad=True, max_pad_num=12)

    # One raw item from each path.
    raw_packed = packed_ds[1]
    raw_padded = padded_ds[1]
    print("\nSingle item (clip 1, text zeros in the first 3 steps)")
    print(f"  packed item: {[tuple(t.shape) if torch.is_tensor(t) else t for t in raw_packed]}")
    print(f"  padded item: {[tuple(t.shape) if torch.is_tensor(t) else t for t in raw_padded]}")
    print(f"  -inf audio became {float(padded_ds[0][1][0, 0]):.1f} (dataset replaces -inf with 0)")

    packed_loader = DataLoader(packed_ds, batch_size=4, shuffle=False, collate_fn=_process_1)
    padded_loader = DataLoader(padded_ds, batch_size=4, shuffle=False, collate_fn=_process_2)
    _show_batch("packed collate (_process_1) — used by LSTM/GRU baselines", next(iter(packed_loader)))
    _show_batch("padded collate (_process_2) — used by GMTM / TransformerEarly", next(iter(padded_loader)))
    print(
        "\nGMTM training always takes the padded path (max_pad=True, T=50 in the"
        " real pickles). Packed lengths exist so LSTM/GRU can skip pad steps."
    )


if __name__ == "__main__":
    main()
