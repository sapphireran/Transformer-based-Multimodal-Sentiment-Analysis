#!/usr/bin/env python3
"""Walk through the MOSI/MOSEI metric bundle on a hand-made example.

Run from the repository root:

    python examples/metrics_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._bootstrap import ensure_output_dir

import numpy as np

from metrics import (
    SENTIMENT_HIGH,
    SENTIMENT_LOW,
    evaluate_sentiment,
    format_score_table,
    split_uniform_5,
    split_uniform_7,
)


def main() -> None:
    # Gold scores span the usual [-3, 3] opinion range, including a zero
    # that the non-zero protocol should drop for Acc-2 / F1.
    y = np.array([-2.8, -1.2, 0.0, 0.4, 1.5, 2.7], dtype=np.float64)

    # A decent regressor: same sign, slightly shrunk, one bin slip.
    y_hat_good = np.array([-2.4, -0.9, 0.1, 0.6, 1.1, 2.2], dtype=np.float64)

    # A collapsed predictor that always emits 0 (common failure mode).
    y_hat_zero = np.zeros_like(y)

    # A sign-flipped predictor.
    y_hat_flip = -y

    rows = {
        "good": evaluate_sentiment(y, y_hat_good),
        "always-zero": evaluate_sentiment(y, y_hat_zero),
        "sign-flip": evaluate_sentiment(y, y_hat_flip),
    }

    print(f"gold labels:      {y}")
    print(f"good predictions: {y_hat_good}")
    print(f"Acc-7 bins gold:  {split_uniform_7(y)}")
    print(f"Acc-7 bins good:  {split_uniform_7(y_hat_good)}")
    print(f"Acc-5 bins gold:  {split_uniform_5(y)}")
    print()
    print(f"bin range is [{SENTIMENT_LOW}, {SENTIMENT_HIGH}]")
    print()
    print(format_score_table(rows))
    print()
    print("notes:")
    print("- Acc-2 / F1 drop the gold-zero clip (index 2) under the non-zero protocol.")
    print("- always-zero has MAE equal to mean(|y|) and undefined-looking Corr (reported as 0).")
    print("- sign-flip keeps MAE of the magnitude error but wrecks Corr and Acc-2.")

    out = ensure_output_dir() / "metrics_demo.md"
    out.write_text(format_score_table(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
