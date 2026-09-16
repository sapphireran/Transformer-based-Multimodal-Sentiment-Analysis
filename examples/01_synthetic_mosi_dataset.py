#!/usr/bin/env python3
"""Build a MOSI-shaped toy pickle and print the schema ``Affectdataset`` expects.

Run from the repo root:

    python examples/01_synthetic_mosi_dataset.py
    python examples/01_synthetic_mosi_dataset.py --full-width
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from common import (
    MOSI_AUDIO_DIM,
    MOSI_BERT_DIM,
    MOSI_SEQ_LEN,
    MOSI_VISION_DIM,
    REPO_ROOT,
    TINY_AUDIO_DIM,
    TINY_SEQ_LEN,
    TINY_TEXT_DIM,
    TINY_VISION_DIM,
    make_synthetic_mosi,
    write_synthetic_pickle,
)


def describe(bundle: dict) -> None:
    print("splits:", ", ".join(bundle.keys()))
    for split_name, split in bundle.items():
        vision = split["vision"]
        print(
            f"  {split_name:5s}  n={vision.shape[0]:3d}  "
            f"vision={tuple(vision.shape[1:])}  "
            f"audio={tuple(split['audio'].shape[1:])}  "
            f"text={tuple(split['text'].shape[1:])}  "
            f"labels={tuple(split['labels'].shape)}  "
            f"y∈[{split['labels'].min():+.2f}, {split['labels'].max():+.2f}]"
        )
        print(f"         first id: {split['id'][0]}")


def try_real_loader(path: Path, data_type: str, max_seq_len: int) -> None:
    """Optional: prove get_dataloader accepts the pickle (needs torch)."""
    sys.path.insert(0, str(REPO_ROOT / "model"))
    from data.get_dataloader import get_dataloader

    train, valid, test = get_dataloader(
        str(path),
        batch_size=4,
        max_seq_len=max_seq_len,
        max_pad=True,
        num_workers=0,
        data_type=data_type,
        train_shuffle=False,
    )
    batch = next(iter(train))
    print(
        "get_dataloader(max_pad=True) batch:",
        f"vision {tuple(batch[0].shape)}",
        f"audio {tuple(batch[1].shape)}",
        f"text {tuple(batch[2].shape)}",
        f"y {tuple(batch[3].shape)}",
    )
    _ = valid, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-width",
        action="store_true",
        help="Use real MOSI BERT widths (50, 35/74/768) instead of the tiny CPU sizes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "examples" / "_generated" / "synthetic_mosi.pkl",
    )
    parser.add_argument("--skip-loader", action="store_true")
    args = parser.parse_args()

    if args.full_width:
        kwargs = dict(
            n_train=12,
            n_valid=4,
            n_test=4,
            seq_len=MOSI_SEQ_LEN,
            vision_dim=MOSI_VISION_DIM,
            audio_dim=MOSI_AUDIO_DIM,
            text_dim=MOSI_BERT_DIM,
        )
        print("Building a full-width MOSI BERT-shaped pickle (still synthetic).")
    else:
        kwargs = dict(
            n_train=24,
            n_valid=8,
            n_test=8,
            seq_len=TINY_SEQ_LEN,
            vision_dim=TINY_VISION_DIM,
            audio_dim=TINY_AUDIO_DIM,
            text_dim=TINY_TEXT_DIM,
        )
        print("Building a tiny MOSI-shaped pickle for CPU examples.")

    bundle = make_synthetic_mosi(**kwargs)
    describe(bundle)
    path = write_synthetic_pickle(args.out, **kwargs)
    print(f"wrote {path} ({path.stat().st_size} bytes)")

    # Round-trip check: the file really is a pickle with the three splits.
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    assert set(loaded) == {"train", "valid", "test"}
    assert loaded["train"]["vision"].shape[0] == kwargs["n_train"]

    if not args.skip_loader:
        try:
            try_real_loader(path, data_type="mosi", max_seq_len=kwargs["seq_len"])
        except Exception as exc:  # pragma: no cover - illustrated in docs
            print(f"loader smoke test skipped: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
