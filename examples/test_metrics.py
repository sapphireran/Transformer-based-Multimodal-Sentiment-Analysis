"""Unit checks for the headless metric helpers and synthetic batches.

Run from the repo root:

    python -m unittest examples.test_metrics
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))

import numpy as np
import torch

from examples.metrics import evaluate_regression, split_uniform, split_uniform_5, split_uniform_7
from examples.shapes import BERT, GLOVE
from examples.synthetic_data import make_batch


class BinningTests(unittest.TestCase):
    def test_endpoints_land_in_first_and_last_bin(self) -> None:
        seven = split_uniform_7([-3.0, 3.0])
        self.assertEqual(list(seven), [1, 7])
        five = split_uniform_5([-3.0, 3.0])
        self.assertEqual(list(five), [1, 5])

    def test_zero_is_a_middle_bin(self) -> None:
        # 7 equal bins on [-3, 3]: zero is the start of bin 4.
        self.assertEqual(int(split_uniform_7([0.0])[0]), 4)
        # 5 equal bins: zero is the start of bin 3.
        self.assertEqual(int(split_uniform_5([0.0])[0]), 3)

    def test_values_outside_range_clip(self) -> None:
        self.assertEqual(int(split_uniform_7([-10.0])[0]), 1)
        self.assertEqual(int(split_uniform_7([10.0])[0]), 7)

    def test_rejects_too_few_bins(self) -> None:
        with self.assertRaises(ValueError):
            split_uniform([0.0], 1)


class ProtocolTests(unittest.TestCase):
    def test_perfect_predictor(self) -> None:
        y = np.array([-2.5, -1.0, 0.2, 1.5, 2.8])
        report = evaluate_regression(y, y)
        self.assertEqual(report.mae, 0.0)
        self.assertEqual(report.acc7, 1.0)
        self.assertEqual(report.acc5, 1.0)
        self.assertEqual(report.acc2, 1.0)
        self.assertEqual(report.f1, 1.0)
        self.assertAlmostEqual(report.corr, 1.0)

    def test_exclude_zero_drops_neutrals(self) -> None:
        y = np.array([0.0, 0.0, 1.0, -1.0])
        y_hat = np.array([0.5, -0.5, 1.0, -1.0])
        dropped = evaluate_regression(y, y_hat, exclude_zero=True)
        kept = evaluate_regression(y, y_hat, exclude_zero=False)
        self.assertEqual(dropped.n_nonzero, 2)
        self.assertEqual(dropped.acc2, 1.0)
        self.assertLess(kept.acc2, 1.0)

    def test_constant_predictor_has_nan_corr(self) -> None:
        y = np.array([-1.0, 0.5, 2.0])
        report = evaluate_regression(y, np.zeros_like(y))
        self.assertTrue(np.isnan(report.corr))


class SyntheticBatchTests(unittest.TestCase):
    def test_bert_shapes(self) -> None:
        batch = make_batch(BERT, batch_size=3, seq_len=9, seed=1)
        self.assertEqual(tuple(batch.vision.shape), (3, 9, 35))
        self.assertEqual(tuple(batch.audio.shape), (3, 9, 74))
        self.assertEqual(tuple(batch.text.shape), (3, 9, 768))
        self.assertEqual(tuple(batch.labels.shape), (3, 1))
        self.assertEqual(batch.widths(), (35, 74, 768))
        self.assertLessEqual(batch.labels.abs().max().item(), 3.0 + 1e-6)

    def test_glove_text_width(self) -> None:
        batch = make_batch(GLOVE, batch_size=2, seq_len=5, seed=2)
        self.assertEqual(batch.text.shape[-1], 300)
        self.assertEqual(GLOVE.early_width, 409)

    def test_zero_modalities_keeps_rank(self) -> None:
        batch = make_batch(BERT, batch_size=2, seq_len=4, seed=3)
        dropped = batch.zero_modalities(["audio", "visual"])
        self.assertTrue(torch.all(dropped.audio == 0))
        self.assertTrue(torch.all(dropped.vision == 0))
        self.assertFalse(torch.all(dropped.text == 0))
        self.assertEqual(tuple(dropped.text.shape), tuple(batch.text.shape))

    def test_unknown_modality_raises(self) -> None:
        batch = make_batch(BERT, batch_size=1, seq_len=2, seed=4)
        with self.assertRaises(KeyError):
            batch.zero_modalities(["smell"])


if __name__ == "__main__":
    unittest.main()
