#!/usr/bin/env python3
"""Show how MAE, Acc-7, and Acc-2 react to bias, noise, and exact zeros.

Perfect predictions are shifted or noised. Acc-7 can drop at a bin wall
while MAE only grows by the shift; Acc-2 stays 1.0 until a sign flip.
Exact-zero labels are optional in binary F1 (``exclude_zero``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from toy_data import make_toy_batch  # noqa: E402

add_model_to_path()

from metrics import compute_sentiment_metrics, format_metrics, split_uniform_7  # noqa: E402


def _shift_table(truth: np.ndarray, shifts: np.ndarray):
    rows = []
    for shift in shifts:
        pred = truth + shift
        scores = compute_sentiment_metrics(truth, pred)
        crossed = int(np.sum(split_uniform_7(pred) != split_uniform_7(truth)))
        rows.append(
            {
                "shift": float(shift),
                "bin_mismatches": crossed,
                **{k: float(v) for k, v in scores.items()},
            }
        )
    return rows


def _print_rows(title: str, rows, keys=("shift", "MAE", "Acc7_uniform", "Acc2", "bin_mismatches")) -> None:
    print(f"== {title} ==")
    header = "  ".join(f"{k:>16}" for k in keys)
    print(header)
    for row in rows:
        cells = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:>16.4f}")
            else:
                cells.append(f"{value:>16}")
        print("  ".join(cells))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    batch = make_toy_batch(batch_size=args.n, seq_len=8, text_backend="glove", seed=args.seed)
    truth = batch.labels.reshape(-1)

    shifts = np.round(np.linspace(-1.2, 1.2, 9), 4)
    shift_rows = _shift_table(truth, shifts)
    _print_rows("additive bias on otherwise-perfect predictions", shift_rows)

    rng = np.random.default_rng(args.seed)
    noise_rows = []
    for scale in (0.0, 0.2, 0.5, 1.0):
        pred = truth + rng.normal(0.0, scale, size=truth.shape)
        scores = compute_sentiment_metrics(truth, pred)
        noise_rows.append({"noise_std": scale, **{k: float(v) for k, v in scores.items()}})
    _print_rows(
        "gaussian noise (new draw per row, same seed stream)",
        noise_rows,
        keys=("noise_std", "MAE", "Acc7_uniform", "Acc2", "Corr"),
    )

    mixed = np.concatenate([truth[: args.n // 2], np.zeros(args.n // 2)])
    pred_pos = np.full_like(mixed, 0.4)
    with_zeros = compute_sentiment_metrics(mixed, pred_pos, exclude_zero=True)
    keep_zeros = compute_sentiment_metrics(mixed, pred_pos, exclude_zero=False)
    print("== exact-zero labels, constant +0.4 prediction ==")
    print("exclude_zero=True")
    print(format_metrics(with_zeros))
    print()
    print("exclude_zero=False")
    print(format_metrics(keep_zeros))
    print()

    # Guard: a 0.3 shift should raise MAE by ~0.3 and should not improve Acc-7.
    zero = next(row for row in shift_rows if abs(row["shift"]) < 1e-9)
    plus = next(row for row in shift_rows if abs(row["shift"] - 0.3) < 1e-9)
    if plus["MAE"] <= zero["MAE"]:
        print("error: positive bias did not increase MAE")
        return 1
    if plus["Acc7_uniform"] > zero["Acc7_uniform"]:
        print("error: positive bias unexpectedly improved Acc-7")
        return 1

    if args.write_json:
        payload = {
            "shifts": shift_rows,
            "noise": noise_rows,
            "exclude_zero_true": {k: float(v) for k, v in with_zeros.items()},
            "exclude_zero_false": {k: float(v) for k, v in keep_zeros.items()},
        }
        out = ensure_output_dir() / "metric_sensitivity.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
