"""Document the aligned pickle schema; optionally validate a file if present."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = ("train", "valid", "test")
REQUIRED_KEYS = ("vision", "audio", "text", "labels")


def schema_description() -> str:
    return """
Aligned pickle (mosei_raw_bert.pkl / mosi_raw_glove.pkl, ...)

dict
├── train / valid / test
│   ├── vision : ndarray [N, T, 35]
│   ├── audio  : ndarray [N, T, 74]     # -inf later replaced by 0
│   ├── text   : ndarray [N, T, 768|300]
│   ├── labels : ndarray [N, 1, ...]    # sentiment in about [-3, 3]
│   └── id     : list[str]              # optional clip ids
""".strip()


def validate_blob(blob: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(blob, dict):
        return ["root is not a dict"]
    for split in REQUIRED_SPLITS:
        if split not in blob:
            errors.append(f"missing split {split!r}")
            continue
        inner = blob[split]
        if not isinstance(inner, dict):
            errors.append(f"{split} is not a dict")
            continue
        for key in REQUIRED_KEYS:
            if key not in inner:
                errors.append(f"{split}.{key} missing")
                continue
            value = inner[key]
            if getattr(value, "ndim", None) is None:
                errors.append(f"{split}.{key} has no ndim")
                continue
            if key != "labels" and value.ndim != 3:
                errors.append(f"{split}.{key} expected rank 3, got {value.ndim}")
        if "vision" in inner and "audio" in inner:
            if inner["vision"].shape[0] != inner["audio"].shape[0]:
                errors.append(f"{split}: vision/audio N mismatch")
    return errors


def maybe_load(path: Path) -> list[str]:
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    return validate_blob(blob)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pickle_path",
        nargs="?",
        help="Optional .pkl to validate. If omitted, print the schema only.",
    )
    args = parser.parse_args(argv)
    print(schema_description())
    print()
    if args.pickle_path:
        path = Path(args.pickle_path)
        if not path.exists():
            print(f"file not found: {path}")
            return 2
        errors = maybe_load(path)
        if errors:
            print("validation failed:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"ok: {path}")
        return 0

    from examples.synthetic_data import make_pickle_dict

    errors = validate_blob(make_pickle_dict())
    if errors:
        raise RuntimeError(errors)
    print("synthetic stand-in validates against the same schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
