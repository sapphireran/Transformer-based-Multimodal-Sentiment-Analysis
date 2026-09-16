"""Unit checks for the personal examples (no MOSI/MOSEI files required)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from common import AUDIO_DIM, TEXT_BERT_DIM, VISUAL_DIM, finite
from eval_protocol import (
    eval_affect,
    regression_and_classification,
    split_uniform_5,
    split_uniform_7,
)
from synthetic_multimodal import (
    SyntheticConfig,
    build_synthetic_dict,
    make_dataloaders,
    random_batch,
    zero_mask,
)


class SyntheticDataTests(unittest.TestCase):
    def test_dict_schema_and_ranges(self):
        cfg = SyntheticConfig(n_train=5, n_valid=3, n_test=3, seq_len=7, text_dim=11)
        data = build_synthetic_dict(cfg)
        self.assertEqual(set(data), {"train", "valid", "test"})
        train = data["train"]
        self.assertEqual(set(train), {"vision", "audio", "text", "labels", "id"})
        self.assertEqual(train["vision"].shape, (5, 7, VISUAL_DIM))
        self.assertEqual(train["audio"].shape, (5, 7, AUDIO_DIM))
        self.assertEqual(train["text"].shape, (5, 7, 11))
        scores = train["labels"][:, 0, 0]
        self.assertTrue(np.all(scores >= -3.0) and np.all(scores <= 3.0))
        self.assertEqual(len(train["id"]), 5)

    def test_zero_mask_matches_ablation_idea(self):
        v, a, t = random_batch(batch_size=2, seq_len=4, text_dim=6)
        mv, ma, mt = zero_mask(v, a, t, keep=("text",))
        self.assertTrue(torch.equal(mt, t))
        self.assertTrue(torch.all(mv == 0))
        self.assertTrue(torch.all(ma == 0))

    def test_dataloader_batch_shapes(self):
        cfg = SyntheticConfig(n_train=10, n_valid=4, n_test=4, seq_len=5, text_dim=9)
        loaders = make_dataloaders(cfg, batch_size=4)
        v, a, t, y = next(iter(loaders["train"]))
        self.assertEqual(v.shape[1:], (5, VISUAL_DIM))
        self.assertEqual(a.shape[1:], (5, AUDIO_DIM))
        self.assertEqual(t.shape[1:], (5, 9))
        self.assertEqual(y.shape[1], 1)


class MetricTests(unittest.TestCase):
    def test_acc7_endpoints(self):
        bins = split_uniform_7([-3.0, 0.0, 3.0])
        self.assertEqual(list(bins), [1, 4, 7])

    def test_acc5_endpoints(self):
        bins = split_uniform_5([-3.0, 0.0, 3.0])
        self.assertEqual(list(bins), [1, 3, 5])

    def test_perfect_predictor(self):
        y = np.array([-2.0, -0.5, 0.2, 1.5, 2.7])
        metrics = regression_and_classification(y, y)
        self.assertAlmostEqual(metrics["MAE"], 0.0, places=6)
        self.assertAlmostEqual(metrics["Acc7_uniform"], 1.0, places=6)
        self.assertAlmostEqual(metrics["Acc2"], 1.0, places=6)
        self.assertAlmostEqual(metrics["Corr"], 1.0, places=6)

    def test_exclude_zero_binary(self):
        y = np.array([-1.0, 0.0, 0.0, 2.0])
        yhat = np.array([-0.2, 1.0, -1.0, 0.4])
        f1, acc = eval_affect(y, yhat, exclude_zero=True)
        # Only the two non-zero gold labels: both signs match → Acc2 = 1.
        self.assertAlmostEqual(acc, 1.0, places=6)
        self.assertGreater(f1, 0.0)

    def test_constant_zero_is_all_negative(self):
        y = np.array([-2.0, -1.0, 1.0, 2.0])
        f1, acc = eval_affect(y, np.zeros_like(y), exclude_zero=True)
        # ŷ > 0 is false for 0.0, so two true negatives and two false
        # negatives on this balanced slice.
        self.assertAlmostEqual(acc, 0.5, places=6)
        self.assertAlmostEqual(f1, 0.0, places=6)


class FusionSmokeTests(unittest.TestCase):
    def test_fusion_zoo_finite(self):
        from demo_fusion import run

        rows = run(verbose=False)
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            self.assertTrue(row["ok"], msg=row["name"])
            self.assertEqual(row["out_shape"][0], 3)

    def test_gmtm_masks_share_params(self):
        from demo_gmtm import run

        rows = run(verbose=False)
        self.assertEqual(len(rows), 7)
        params = {row["params"] for row in rows.values()}
        self.assertEqual(len(params), 1)
        self.assertTrue(all(row["ok"] for row in rows.values()))


class LoggedResultsTests(unittest.TestCase):
    def test_all_csvs_present_and_best_mae(self):
        from print_logged_results import run

        rows = run(verbose=False)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["ok"] for r in rows))
        mosei_bert = next(r for r in rows if r["title"] == "MOSEI BERT fusion")
        self.assertEqual(mosei_bert["best"], "TransformerLate")
        self.assertEqual(mosei_bert["n"], 6)

    def test_format_table_has_header_rule(self):
        from print_logged_results import format_table

        text = format_table([["Fusion", "MAE"], ["A", "0.2"], ["B", "0.1"]])
        self.assertIn("Fusion", text)
        self.assertIn("---", text)


class ShapeTests(unittest.TestCase):
    def test_bert_widths(self):
        self.assertEqual(VISUAL_DIM + AUDIO_DIM + TEXT_BERT_DIM, 877)

    def test_tfn_width_recipe(self):
        # train_main_bert.py uses 19 / 39 / 159 → +1 each for the TFN bias.
        self.assertEqual((19 + 1) * (39 + 1) * (159 + 1), 128000)

    def test_modules_accept_random_batch(self):
        import models as M

        v, a, t = random_batch(batch_size=2, seq_len=5, text_dim=32)
        out = M.ConcatEarly()([v, a, t])
        self.assertEqual(out.shape[-1], VISUAL_DIM + AUDIO_DIM + 32)
        self.assertTrue(finite(out))


if __name__ == "__main__":
    unittest.main()
