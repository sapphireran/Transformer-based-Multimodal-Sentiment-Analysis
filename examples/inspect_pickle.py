"""Print split sizes and feature widths if a local MOSI/MOSEI pickle exists.

The pickles are not in git. When they are absent the script exits 0
after saying so, so ``run_all.py`` can ignore this file.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import _paths
from _paths import MODEL_DIR

CANDIDATES = [
    MODEL_DIR / "data" / "MOSEI" / "mosei_raw_bert.pkl",
    MODEL_DIR / "data" / "MOSEI" / "mosei_raw_glove.pkl",
    MODEL_DIR / "data" / "MOSI" / "mosi_raw_bert.pkl",
    MODEL_DIR / "data" / "MOSI" / "mosi_raw_glove.pkl",
]


def describe(path: Path) -> str:
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    lines = [f"# {path.relative_to(_paths.REPO_ROOT)}"]
    if not isinstance(blob, dict):
        return lines[0] + f"\nunexpected type {type(blob)}"
    for split in ("train", "valid", "test"):
        if split not in blob:
            lines.append(f"  {split}: missing")
            continue
        block = blob[split]
        n = None
        bits = []
        for key in ("vision", "audio", "text", "labels"):
            if key not in block:
                bits.append(f"{key}=?")
                continue
            arr = np.asarray(block[key])
            bits.append(f"{key}{tuple(arr.shape)}")
            n = arr.shape[0]
        lines.append(f"  {split} N={n}: " + ", ".join(bits))
    return "\n".join(lines)


def found_pickles() -> list[Path]:
    return [path for path in CANDIDATES if path.is_file() and path.stat().st_size > 0]


def _demo() -> list[Path]:
    present = found_pickles()
    if not present:
        print("inspect_pickle: no local MOSI/MOSEI pickles "
              "(this is expected in a fresh clone)")
        return []
    for path in present:
        print(describe(path))
    return present


if __name__ == "__main__":
    _demo()
