#!/usr/bin/env python3
"""Show Acc7 / Acc5 / Acc2 / F1 on a hand-picked score sheet.

The numbers are small enough to check by hand against
``docs/evaluation.md``. Used as both a demo and a regression fixture.
"""

from __future__ import annotations

import json

from synthetic_data import ensure_model_on_path

ensure_model_on_path()

import numpy as np
from sklearn.metrics import accuracy_score
from train_and_test import eval_affect, split_uniform_5, split_uniform_7


# Gold scores chosen to land in different uniform bins, including a zero.
Y = np.array([-2.8, -1.5, -0.4, 0.0, 0.6, 1.7, 2.9], dtype=np.float64)
# Predictions: a few exact, a few that cross a boundary, one sign flip.
YHAT = np.array([-2.6, -1.1, 0.2, 0.1, 0.55, 2.4, 2.7], dtype=np.float64)


def bin_table(values: np.ndarray) -> list[dict]:
    c7 = split_uniform_7(values)
    c5 = split_uniform_5(values)
    rows = []
    for v, a, b in zip(values.tolist(), c7.tolist(), c5.tolist()):
        rows.append({"value": v, "acc7_bin": int(a), "acc5_bin": int(b)})
    return rows


def run() -> dict:
    acc7 = float(accuracy_score(split_uniform_7(Y), split_uniform_7(YHAT)))
    acc5 = float(accuracy_score(split_uniform_5(Y), split_uniform_5(YHAT)))
    f1, acc2 = eval_affect(Y, YHAT, exclude_zero=True)
    mae = float(np.mean(np.abs(Y - YHAT)))
    return {
        "gold": bin_table(Y),
        "pred": bin_table(YHAT),
        "mae": round(mae, 4),
        "acc7": round(acc7, 4),
        "acc5": round(acc5, 4),
        "acc2": round(float(acc2), 4),
        "f1": round(float(f1), 4),
        "notes": {
            "zero_gold_dropped_from_acc2": True,
            "sign_flip_at_index": 2,
            "n_binary": int((Y != 0).sum()),
        },
    }


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
