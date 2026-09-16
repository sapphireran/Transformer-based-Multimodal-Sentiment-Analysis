#!/usr/bin/env python3
"""Assertions for the personal docs/examples helpers.

Torch-dependent checks are skipped when PyTorch is not installed so the
CSV / metric documentation can still be verified in a bare interpreter.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.lib.metrics import (
    acc2_f1,
    classification_scores,
    mae,
    split_uniform_5,
    split_uniform_7,
    uniform7_edges,
)
from examples.lib.repo import RESULT_CSV_PATHS
from examples.lib.synthetic import feature_widths, make_tiny_affect_dict
from examples.tiny_affect_dataset import TinyAffectDataset


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class MetricTests(unittest.TestCase):
    def test_acc7_edges_are_equal_width(self):
        edges = uniform7_edges()
        self.assertEqual(len(edges), 8)
        self.assertAlmostEqual(edges[0], -3.0)
        self.assertAlmostEqual(edges[-1], 3.0)
        widths = [edges[i + 1] - edges[i] for i in range(7)]
        for width in widths:
            self.assertAlmostEqual(width, 6.0 / 7.0, places=6)

    def test_near_neutral_values_split_across_acc7_bins(self):
        # Boundary is -3 + 4*(6/7) ≈ 0.4286, so 0.40 and 0.50 differ.
        bins = split_uniform_7([0.40, 0.50])
        self.assertNotEqual(int(bins[0]), int(bins[1]))

    def test_acc7_endpoints_clip(self):
        bins = split_uniform_7([-3.0, 3.0])
        self.assertEqual(int(bins[0]), 1)
        self.assertEqual(int(bins[1]), 7)

    def test_acc5_endpoints(self):
        bins = split_uniform_5([-3.0, 3.0])
        self.assertEqual(int(bins[0]), 1)
        self.assertEqual(int(bins[1]), 5)

    def test_mae_hand_check(self):
        self.assertAlmostEqual(mae([-2.0, -0.2, 0.0, 0.8, 2.4], [-1.5, 0.3, 0.1, 0.9, 2.0]), 0.32)

    def test_acc2_drops_exact_zero_gold(self):
        f1, acc2 = acc2_f1([-2.0, -0.2, 0.0, 0.8, 2.4], [-1.5, 0.3, 0.1, 0.9, 2.0])
        self.assertAlmostEqual(acc2, 0.75)
        self.assertGreater(f1, 0.0)

    def test_perfect_prediction_scores(self):
        values = [-2.0, -1.0, 0.5, 1.5, 2.5]
        clf = classification_scores(values, values)
        self.assertEqual(clf["Acc7_uniform"], 1.0)
        self.assertEqual(clf["Acc5_uniform"], 1.0)
        self.assertEqual(clf["Acc2"], 1.0)
        self.assertEqual(clf["F1"], 1.0)


class CsvTests(unittest.TestCase):
    def test_all_result_tables_exist_and_have_mae(self):
        for key, path in RESULT_CSV_PATHS.items():
            with self.subTest(key=key):
                self.assertTrue(path.exists(), path)
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreaterEqual(len(rows), 1)
                header = {name.strip().upper() for name in rows[0].keys()}
                self.assertTrue("MAE" in header)
                self.assertTrue(any(name.startswith("ACC") for name in header))

    def test_mosei_bert_transformer_late_is_best_mae(self):
        path = RESULT_CSV_PATHS["mosei_bert_fusion"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        best = min(rows, key=lambda row: float(row["MAE"]))
        self.assertEqual(best["Fusion Method"], "TransformerLate")
        self.assertAlmostEqual(float(best["MAE"]), 0.5846)

    def test_mosei_bert_gmtm_full_trio_mae(self):
        path = RESULT_CSV_PATHS["mosei_bert_gmtm"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        full = [row for row in rows if "visual" in row["Fusion Method"] and "text" in row["Fusion Method"] and "audio" in row["Fusion Method"]]
        self.assertEqual(len(full), 1)
        self.assertAlmostEqual(float(full[0]["MAE"]), 0.5640)


class TinyDatasetTests(unittest.TestCase):
    def test_pickle_schema_and_nonzero_text(self):
        blob = make_tiny_affect_dict(n_train=5, n_valid=3, n_test=3, embedding="glove", seed=2)
        self.assertEqual(set(blob), {"train", "valid", "test"})
        widths = feature_widths("glove")
        for split in blob.values():
            self.assertEqual(set(split), {"vision", "audio", "text", "labels", "id"})
            self.assertEqual(split["vision"].shape[2], widths["visual"])
            self.assertEqual(split["audio"].shape[2], widths["audio"])
            self.assertEqual(split["text"].shape[2], widths["text"])
            self.assertEqual(split["labels"].ndim, 3)
            self.assertNotEqual(float(split["text"].sum()), 0.0)

    def test_dataset_alignment_and_label_rank(self):
        blob = make_tiny_affect_dict(n_train=4, embedding="bert", seed=3)
        dataset = TinyAffectDataset(blob["train"])
        self.assertEqual(len(dataset), 4)
        vision, audio, text, label = dataset[0]
        self.assertEqual(vision.ndim, 2)
        self.assertEqual(label.shape, (1,))
        self.assertEqual(vision.shape[1], 35)
        self.assertEqual(audio.shape[1], 74)
        self.assertEqual(text.shape[1], 768)


@unittest.skipUnless(_torch_available(), "PyTorch is not installed")
class TorchExampleTests(unittest.TestCase):
    def test_gmtm_forward_shape(self):
        from examples.lib.model_import import load_models
        from examples.lib.synthetic import make_mosei_like_batch

        class Params:
            num_heads = 4
            layers = 1
            attn_dropout = 0.0
            attn_dropout_modalities = [0.0, 0.0, 0.0]
            relu_dropout = 0.0
            res_dropout = 0.0
            out_dropout = 0.0
            embed_dropout = 0.0
            embed_dim = 16
            attn_mask = False
            output_dim = 1
            all_steps = False

        models = load_models()
        vision, audio, text, _ = make_mosei_like_batch(batch_size=2, seq_len=5, seed=9)
        model = models.GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=Params)
        model.eval()
        import torch

        with torch.no_grad():
            out = model([vision, audio, text])
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_concat_and_tensor_fusion_shapes(self):
        import torch

        from examples.lib.model_import import load_models

        models = load_models()
        streams = [torch.randn(3, 5, 8), torch.randn(3, 5, 12), torch.randn(3, 5, 16)]
        early = models.ConcatEarly()(streams)
        self.assertEqual(tuple(early.shape), (3, 5, 36))
        late = models.ConcatLate()(streams)
        self.assertEqual(tuple(late.shape), (3, 5 * 8 + 5 * 12 + 5 * 16))
        fused = models.TensorFusion()([s.mean(1) for s in streams])
        self.assertEqual(tuple(fused.shape), (3, 9 * 13 * 17))

    def test_toy_train_drops_mae(self):
        from examples.toy_train_gmtm import train

        report = train(steps=20, batch_size=20, seq_len=6, seed=11)
        self.assertLess(report["last_mae"], report["first_mae"] - 0.10)


class DocsExistTests(unittest.TestCase):
    def test_doc_pages_are_present(self):
        docs = REPO_ROOT / "docs"
        expected = [
            "README.md",
            "setup.md",
            "architecture.md",
            "datasets.md",
            "metrics.md",
            "experiments.md",
            "results.md",
            "code_map.md",
            "personal_lab_notes.md",
        ]
        for name in expected:
            self.assertTrue((docs / name).exists(), name)
        self.assertIn("Gated Multi-Transformer", (docs / "architecture.md").read_text())
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertIn("examples/", readme)
        self.assertIn("docs/", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
