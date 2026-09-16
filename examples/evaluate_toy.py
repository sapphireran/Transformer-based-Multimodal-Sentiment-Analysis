#!/usr/bin/env python3
"""Score synthetic labels with the same helpers the training loop uses."""

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


def _oracle_from_text(batch) -> np.ndarray:
    """Invert the text bias written by ``make_toy_batch`` (cue scale 0.90)."""
    text_score = batch.text.mean(axis=(1, 2))
    pred = np.clip(text_score / 0.90 * 3.0, -3.0, 3.0)
    return pred.reshape(-1)


def _constant_baseline(batch) -> np.ndarray:
    return np.zeros(batch.batch_size, dtype=np.float64)


def _noisy_copy(batch, rng: np.random.Generator, scale: float = 0.4) -> np.ndarray:
    return (batch.labels.reshape(-1) + rng.normal(0.0, scale, size=batch.batch_size))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--backend", choices=("bert", "glove"), default="bert")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    batch = make_toy_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_backend=args.backend,
        seed=args.seed,
    )
    truth = batch.labels.reshape(-1)
    rng = np.random.default_rng(args.seed + 1)

    reports = {
        "oracle_text_mean": compute_sentiment_metrics(truth, _oracle_from_text(batch)),
        "noisy_copy": compute_sentiment_metrics(truth, _noisy_copy(batch, rng)),
        "constant_zero": compute_sentiment_metrics(truth, _constant_baseline(batch)),
    }

    print(f"Toy batch  backend={args.backend}  N={args.batch_size}  T={args.seq_len}")
    print(f"Label range [{truth.min():.2f}, {truth.max():.2f}]")
    print(f"Acc-7 bins used by the labels: {sorted(set(split_uniform_7(truth)))}")
    print()
    for name, scores in reports.items():
        print(f"== {name} ==")
        print(format_metrics(scores))
        print()

    if args.write_json:
        out = ensure_output_dir() / "evaluate_toy.json"
        serializable = {
            name: {k: float(v) for k, v in scores.items()} for name, scores in reports.items()
        }
        out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    # Guard rail: the text-mean oracle should beat a constant-0 predictor on MAE.
    if reports["oracle_text_mean"]["MAE"] >= reports["constant_zero"]["MAE"]:
        print("warning: text-mean oracle did not beat the constant baseline on this seed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
