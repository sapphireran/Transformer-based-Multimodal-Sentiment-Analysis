#!/usr/bin/env python3
"""Show the two collate signatures the real dataloader switches between.

``max_pad=False`` → packed features + lengths (LSTM / GRU path).
``max_pad=True``  → stacked ``(B, T, F)`` (GMTM / TransformerEarly).
"""

from __future__ import annotations

import json

from synthetic_data import (
    FeatureSpec,
    collate_packed,
    collate_padded,
    random_variable_lengths,
    seed_everything,
)


def _shape(x):
    if isinstance(x, list):
        return [_shape(v) for v in x]
    return list(x.shape)


def run() -> dict:
    seed_everything(9)
    spec = FeatureSpec(seq_len=12)
    rows = random_variable_lengths(batch_size=5, spec=spec, min_len=4, seed=9)
    lengths = [int(row[0].size(0)) for row in rows]

    packed = collate_packed(rows)
    features, feat_lengths, indices, packed_labels = packed

    padded = collate_padded(rows, max_len=spec.seq_len)
    vision, audio, text, padded_labels = padded

    return {
        "sample_lengths": lengths,
        "packed": {
            "feature_shapes": _shape(features),
            "length_shapes": _shape(feat_lengths),
            "indices": indices.view(-1).tolist(),
            "label_shape": _shape(packed_labels),
            "max_T": [int(f.size(1)) for f in features],
        },
        "padded": {
            "vision": _shape(vision),
            "audio": _shape(audio),
            "text": _shape(text),
            "label_shape": _shape(padded_labels),
            "T": spec.seq_len,
        },
        "same_label_count": packed_labels.size(0) == padded_labels.size(0),
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
