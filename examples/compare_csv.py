#!/usr/bin/env python3
"""Diff a fresh sweep CSV against the snapshot in ``model/results/``.

This does not prove a retrain is correct. It catches the boring
failures: a method disappeared, a column was renamed, or MAE jumped
by more than ``--mae-tol``. Exit status is 1 when any row is flagged.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = ("Fusion Method", "MAE")


def _load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: empty CSV")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}; have {reader.fieldnames}")
        rows = {}
        for row in reader:
            key = row["Fusion Method"].strip()
            if not key:
                continue
            rows[key] = row
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    return rows


def _as_float(row: dict[str, str], column: str) -> float | None:
    raw = row.get(column)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="CSV checked into model/results/")
    parser.add_argument("--fresh", type=Path, required=True, help="CSV just written by a training script")
    parser.add_argument("--mae-tol", type=float, default=0.03, help="absolute MAE drift allowed per row")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=["MAE", "ACC7", "Acc5", "ACC2", "Corr", "F1"],
        help="numeric columns to print (MAE is the one that fails the run)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.snapshot.is_file():
        raise SystemExit(f"snapshot not found: {args.snapshot}")
    if not args.fresh.is_file():
        raise SystemExit(f"fresh CSV not found: {args.fresh}")

    snap = _load(args.snapshot)
    fresh = _load(args.fresh)

    print(f"snapshot {args.snapshot}  ({len(snap)} rows)")
    print(f"fresh    {args.fresh}  ({len(fresh)} rows)")
    print(f"mae-tol  {args.mae_tol}")

    flags = 0
    missing = sorted(set(snap) - set(fresh))
    extra = sorted(set(fresh) - set(snap))
    if missing:
        print(f"\nmissing from fresh: {missing}")
        flags += len(missing)
    if extra:
        print(f"extra in fresh:     {extra}")

    print("\n{:<32} {:>8} {:>8} {:>8}  notes".format("method", "snap MAE", "new MAE", "delta"))
    for name in snap:
        if name not in fresh:
            continue
        old = _as_float(snap[name], "MAE")
        new = _as_float(fresh[name], "MAE")
        if old is None or new is None:
            note = "non-numeric MAE"
            flags += 1
            delta = float("nan")
        else:
            delta = new - old
            note = ""
            if abs(delta) > args.mae_tol:
                note = f"MAE drift > {args.mae_tol}"
                flags += 1
        extras = []
        for column in args.compare:
            if column == "MAE":
                continue
            a = _as_float(snap[name], column)
            b = _as_float(fresh[name], column)
            if a is not None and b is not None:
                extras.append(f"{column} {b - a:+.4f}")
        extra_txt = ("  " + ", ".join(extras)) if extras else ""
        print(f"{name:<32} {old if old is not None else float('nan'):>8.4f} "
              f"{new if new is not None else float('nan'):>8.4f} {delta:>+8.4f}  {note}{extra_txt}")

    if flags:
        print(f"\n{flags} flag(s).")
        return 1
    print("\nNo flags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
