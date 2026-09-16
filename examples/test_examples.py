"""Unittest checks for the synthetic examples.

Run from the repository root:

    python examples/test_examples.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from common import BERT_EARLY_DIM, MAX_SEQ_LEN, feature_dims  # noqa: E402
from metrics_demo import evaluate, split_uniform_5, split_uniform_7  # noqa: E402
from synthetic_data import make_corpus  # noqa: E402


class SyntheticDataTests(unittest.TestCase):
    def test_bert_padded_shapes(self):
        corpus = make_corpus(n=5, text="bert", seed=11)
        v, a, t, y = corpus.padded()
        fv, fa, ft = feature_dims("bert")
        self.assertEqual(tuple(v.shape), (5, MAX_SEQ_LEN, fv))
        self.assertEqual(tuple(a.shape), (5, MAX_SEQ_LEN, fa))
        self.assertEqual(tuple(t.shape), (5, MAX_SEQ_LEN, ft))
        self.assertEqual(tuple(y.shape), (5, 1))
        self.assertEqual(fv + fa + ft, BERT_EARLY_DIM)

    def test_glove_text_width(self):
        corpus = make_corpus(n=3, text="glove", seed=12)
        _, _, t, _ = corpus.padded()
        self.assertEqual(t.shape[-1], feature_dims("glove")[2])

    def test_zeroing_leaves_text(self):
        corpus = make_corpus(n=4, text="bert", seed=13)
        v, a, t, _ = corpus.zero_modalities({"text"})
        self.assertEqual(v.abs().sum().item(), 0.0)
        self.assertEqual(a.abs().sum().item(), 0.0)
        self.assertGreater(t.abs().sum().item(), 0.0)

    def test_packed_lengths_match(self):
        corpus = make_corpus(n=4, text="bert", min_len=7, max_len=15, seed=14)
        features, lengths, labels = corpus.packed()
        self.assertEqual(len(features), 3)
        self.assertEqual(len(lengths), 3)
        self.assertEqual(tuple(labels.shape), (4, 1))
        self.assertTrue(torch.equal(lengths[0], corpus.lengths))
        self.assertEqual(features[0].shape[1], int(corpus.lengths.max().item()))

    def test_labels_in_range(self):
        corpus = make_corpus(n=16, text="bert", seed=15)
        self.assertGreaterEqual(corpus.labels.min().item(), -3.0)
        self.assertLessEqual(corpus.labels.max().item(), 3.0)
        # Latent sentiment is drawn on [-2.6, 2.6]; a batch of 16 should
        # not collapse near zero the way a raw feature-mean used to.
        self.assertGreater(corpus.labels.max().item() - corpus.labels.min().item(), 1.5)


class MetricsTests(unittest.TestCase):
    def test_perfect_scores(self):
        y = np.array([-2.0, -1.0, 1.0, 2.0])
        m = evaluate(y, y)
        self.assertAlmostEqual(m["MAE"], 0.0, places=6)
        self.assertAlmostEqual(m["Acc2"], 1.0, places=6)
        self.assertAlmostEqual(m["F1"], 1.0, places=6)
        self.assertAlmostEqual(m["Corr"], 1.0, places=6)
        self.assertAlmostEqual(m["Acc7_uniform"], 1.0, places=6)
        self.assertAlmostEqual(m["Acc5_uniform"], 1.0, places=6)

    def test_exclude_zero_from_acc2(self):
        y_true = np.array([0.0, 0.0, 1.5, -1.5])
        y_pred = np.array([9.0, -9.0, 1.0, -1.0])
        m = evaluate(y_true, y_pred)
        # The two zeros are ignored; the two nonzero items are correct.
        self.assertAlmostEqual(m["Acc2"], 1.0, places=6)

    def test_uniform_clip_ends(self):
        self.assertEqual(int(split_uniform_7(np.array([3.0]))[0]), 7)
        self.assertEqual(int(split_uniform_7(np.array([-3.0]))[0]), 1)
        self.assertEqual(int(split_uniform_5(np.array([3.0]))[0]), 5)
        self.assertEqual(int(split_uniform_5(np.array([-3.0]))[0]), 1)


class ResultTableTests(unittest.TestCase):
    def test_all_csvs_parse(self):
        from result_tables import TABLES, format_table

        for title, path in TABLES:
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            text = format_table(title, path)
            self.assertIn("*", text)
            self.assertIn(title, text)

    def test_mosei_bert_mae_winner_is_transformer_late(self):
        from result_tables import REPO_ROOT, _best_index, _read, _as_float

        header, body = _read(REPO_ROOT / "model" / "results" / "main_results.csv")
        mae_col = header.index("MAE")
        values = [_as_float(row[mae_col]) for row in body]
        winner = _best_index(values, minimize=True)
        self.assertEqual(body[winner][0], "TransformerLate")


class ModelImportTests(unittest.TestCase):
    def test_concat_and_lmf_forward(self):
        from models import ConcatEarly, LowRankTensorFusion

        corpus = make_corpus(n=2, text="bert", min_len=6, max_len=8, seed=16)
        v, a, t, _ = corpus.padded(max_len=8)
        y = ConcatEarly()([v, a, t])
        self.assertEqual(y.shape[-1], BERT_EARLY_DIM)
        lmf = LowRankTensorFusion([8, 8, 8], 4, 2)
        out = lmf([torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8)])
        self.assertEqual(tuple(out.shape), (2, 4))

    def test_gmtm_forward_cpu(self):
        from gmtm_toy_train import ToyHParams
        from models import GatedMultiTransfomerModel

        dims = list(feature_dims("bert"))
        model = GatedMultiTransfomerModel(3, dims, hyp_params=ToyHParams)
        corpus = make_corpus(n=2, text="bert", min_len=6, max_len=8, seed=17)
        v, a, t, _ = corpus.padded(max_len=8)
        pred = model([v, a, t])
        self.assertEqual(pred.reshape(2, -1).shape[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
