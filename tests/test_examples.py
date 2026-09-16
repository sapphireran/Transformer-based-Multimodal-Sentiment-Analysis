"""Tests for the synthetic example helpers and (when torch is present) models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(MODEL_DIR))

from toy_data import (  # noqa: E402
    AUDIO_DIM,
    BERT_DIM,
    GLOVE_DIM,
    VISION_DIM,
    describe_batch,
    make_toy_batch,
    make_toy_split,
)
from inspect_results import load_table  # noqa: E402
from metrics import compute_sentiment_metrics  # noqa: E402

torch = pytest.importorskip("torch", reason="fusion / GMTM examples need torch")


class TestToyData:
    def test_bert_shapes(self):
        batch = make_toy_batch(batch_size=5, seq_len=9, text_backend="bert", seed=1)
        assert batch.vision.shape == (5, 9, VISION_DIM)
        assert batch.audio.shape == (5, 9, AUDIO_DIM)
        assert batch.text.shape == (5, 9, BERT_DIM)
        assert batch.labels.shape == (5, 1)
        assert batch.labels.min() >= -3.0
        assert batch.labels.max() <= 3.0

    def test_labels_span_both_signs(self):
        batch = make_toy_batch(batch_size=48, seq_len=8, seed=0)
        assert batch.labels.min() < -1.0
        assert batch.labels.max() > 1.0

    def test_glove_text_width(self):
        batch = make_toy_batch(batch_size=3, seq_len=4, text_backend="glove")
        assert batch.text.shape[-1] == GLOVE_DIM

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError):
            make_toy_batch(text_backend="word2vec")

    def test_mask_zeros_dropped_modality(self):
        batch = make_toy_batch(batch_size=2, seq_len=4, seed=2)
        masked = batch.masked(("text",))
        assert np.allclose(masked.vision, 0)
        assert np.allclose(masked.audio, 0)
        assert not np.allclose(masked.text, 0)
        assert masked.modalities == ("text",)

    def test_split_counts(self):
        splits = make_toy_split(n_train=10, n_valid=4, n_test=5, batch_size=4, seq_len=6)
        assert sum(b.batch_size for b in splits["train"]) == 10
        assert sum(b.batch_size for b in splits["valid"]) == 4
        assert sum(b.batch_size for b in splits["test"]) == 5

    def test_describe_contains_backend(self):
        text = describe_batch(make_toy_batch(batch_size=2, seq_len=3, text_backend="glove"))
        assert "glove" in text
        assert "vision" in text


class TestInspectResults:
    def test_main_results_has_six_fusion_rows(self):
        rows = load_table(MODEL_DIR / "main_results.csv")
        methods = [row["Fusion Method"] for row in rows]
        assert "TransformerLate" in methods
        assert len(rows) == 6

    def test_ablation_method_names_are_cleaned(self):
        rows = load_table(MODEL_DIR / "ablation_results.csv")
        methods = {row["Fusion Method"] for row in rows}
        assert "text+audio+visual" in methods
        assert "text" in methods


class TestFusionAndGMTM:
    def test_concat_and_tensor_fusion_shapes(self):
        from models import ConcatEarly, ConcatLate, TensorFusion

        batch = make_toy_batch(batch_size=3, seq_len=7, text_backend="glove", seed=4)
        vision = torch.from_numpy(batch.vision)
        audio = torch.from_numpy(batch.audio)
        text = torch.from_numpy(batch.text)

        early = ConcatEarly()([vision, audio, text])
        assert early.shape == (3, 7, VISION_DIM + AUDIO_DIM + GLOVE_DIM)

        late = ConcatLate()([vision.mean(1), audio.mean(1), text.mean(1)])
        assert late.shape == (3, VISION_DIM + AUDIO_DIM + GLOVE_DIM)

        fused = TensorFusion()(
            [vision.mean(1)[:, :4], audio.mean(1)[:, :4], text.mean(1)[:, :4]]
        )
        # (4+1) * (4+1) * (4+1) after the ones-augmented outer products
        assert fused.shape[-1] == 125

    def test_gmtm_forward_is_finite(self):
        from gmtm_forward import TinyHParams
        from models import GatedMultiTransfomerModel

        batch = make_toy_batch(batch_size=2, seq_len=6, text_backend="glove", seed=8)
        model = GatedMultiTransfomerModel(
            3, [VISION_DIM, AUDIO_DIM, GLOVE_DIM], hyp_params=TinyHParams
        )
        model.eval()
        with torch.no_grad():
            pred = model(
                [
                    torch.from_numpy(batch.vision),
                    torch.from_numpy(batch.audio),
                    torch.from_numpy(batch.text),
                ]
            )
        assert pred.shape == (2, 1)
        assert torch.isfinite(pred).all()

    def test_tiny_train_step_descends(self):
        from gmtm_forward import TinyHParams
        from models import GatedMultiTransfomerModel

        torch.manual_seed(0)
        batch = make_toy_batch(batch_size=8, seq_len=8, text_backend="glove", seed=0)
        model = GatedMultiTransfomerModel(
            3, [VISION_DIM, AUDIO_DIM, GLOVE_DIM], hyp_params=TinyHParams
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.L1Loss()
        vision = torch.from_numpy(batch.vision)
        audio = torch.from_numpy(batch.audio)
        text = torch.from_numpy(batch.text)
        y = torch.from_numpy(batch.labels)

        model.train()
        pred0 = model([vision, audio, text])
        loss0 = loss_fn(pred0, y)
        loss0.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        pred1 = model([vision, audio, text])
        loss1 = loss_fn(pred1, y)
        assert torch.isfinite(loss0) and torch.isfinite(loss1)

    def test_metrics_on_model_output(self):
        from gmtm_forward import TinyHParams
        from models import GatedMultiTransfomerModel

        batch = make_toy_batch(batch_size=6, seq_len=5, text_backend="glove", seed=9)
        model = GatedMultiTransfomerModel(
            3, [VISION_DIM, AUDIO_DIM, GLOVE_DIM], hyp_params=TinyHParams
        )
        model.eval()
        with torch.no_grad():
            pred = model(
                [
                    torch.from_numpy(batch.vision),
                    torch.from_numpy(batch.audio),
                    torch.from_numpy(batch.text),
                ]
            )
        scores = compute_sentiment_metrics(batch.labels, pred)
        assert set(scores) == {
            "MAE",
            "MSE",
            "Corr",
            "Acc7_uniform",
            "Acc5_uniform",
            "Acc2",
            "F1",
        }
        assert scores["MAE"] >= 0.0


class TestPlotAndCollate:
    def test_with_gmtm_appends_renamed_row(self):
        from plot_results import metric_value, with_gmtm

        fusion = load_table(MODEL_DIR / "main_results.csv")
        ablation = load_table(MODEL_DIR / "ablation_results.csv")
        merged = with_gmtm(fusion, ablation)
        assert len(merged) == len(fusion) + 1
        assert merged[-1]["Fusion Method"] == "GatedMultiTransformer"
        assert metric_value(merged[-1], "MAE") == pytest.approx(0.5640)

    def test_plot_results_writes_pngs(self, tmp_path):
        from plot_results import plot_fusion_and_ablation

        paths = plot_fusion_and_ablation(tmp_path)
        assert len(paths) == 3
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_packed_t_max_matches_longest_clip(self):
        from packed_vs_padded import collate_packed, make_variable_clips

        clips = make_variable_clips(batch_size=6, max_len=10, seed=3)
        packed, lengths, labels = collate_packed(clips)
        raw = [clip["length"] for clip in clips]
        assert packed[0].shape[0] == 6
        assert packed[0].shape[1] == max(raw)
        assert lengths[0].tolist() == raw
        assert labels.shape[0] == 6

    def test_padded_is_fixed_t(self):
        from packed_vs_padded import collate_padded, make_variable_clips

        clips = make_variable_clips(batch_size=4, max_len=9, seed=1)
        cubes, labels = collate_padded(clips, max_len=9)
        assert cubes[0].shape == (4, 9, VISION_DIM)
        assert cubes[1].shape == (4, 9, AUDIO_DIM)
        assert cubes[2].shape == (4, 9, GLOVE_DIM)
        assert labels.shape[0] == 4


class TestCountParams:
    def test_tiny_gmtm_is_smaller_than_paper(self):
        from count_params import describe_zoo

        rows = {row["name"]: row["parameters"] for row in describe_zoo(GLOVE_DIM)}
        assert rows["GMTM TinyHParams"] < rows["GMTM paper HParams"]
        assert rows["GMTM TinyHParams"] > 0
        assert rows["TensorFusion+head"] > 1_000_000

