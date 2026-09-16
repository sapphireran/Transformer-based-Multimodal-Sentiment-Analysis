"""Tests for the personal docs/examples — no CMU data, no checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"

if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from common import (  # noqa: E402
    AUDIO_DIM,
    BERT_DIM,
    GLOVE_DIM,
    SEQ_LEN,
    VISUAL_DIM,
    make_synthetic_batch,
    zero_modalities,
)
from fusion_forward import EXPECTED_LAST_DIM, run_all  # noqa: E402
from gmtm_forward import run_one  # noqa: E402
from metrics_demo import Y, YHAT, metrics_by_hand, metrics_from_repo  # noqa: E402
from summarize_results import TABLES, load_table  # noqa: E402


def test_synthetic_batch_shapes_bert():
    batch = make_synthetic_batch(batch_size=3, embedding="bert", seed=1)
    assert batch.vision.shape == (3, SEQ_LEN, VISUAL_DIM)
    assert batch.audio.shape == (3, SEQ_LEN, AUDIO_DIM)
    assert batch.text.shape == (3, SEQ_LEN, BERT_DIM)
    assert batch.labels.shape == (3, 1)
    assert float(batch.labels.min()) >= -3.0
    assert float(batch.labels.max()) <= 3.0
    assert batch.early_width == 877


def test_synthetic_batch_shapes_glove():
    batch = make_synthetic_batch(batch_size=2, embedding="glove", seed=2)
    assert batch.text.shape[-1] == GLOVE_DIM
    assert batch.early_width == 409


def test_zero_modalities_matches_ablation_contract():
    batch = make_synthetic_batch(batch_size=2, embedding="bert", seed=3)
    vis, aud, txt = zero_modalities(batch, ("text",))
    assert torch.count_nonzero(vis) == 0
    assert torch.count_nonzero(aud) == 0
    assert torch.equal(txt, batch.text)
    with pytest.raises(ValueError):
        zero_modalities(batch, ("smell",))


def test_fusion_forward_shapes_and_finite():
    rows = run_all(batch_size=2, seed=0)
    names = {row["fusion"] for row in rows}
    assert names == set(EXPECTED_LAST_DIM)
    for row in rows:
        assert row["finite"] is True
        assert row["output"].size(-1) == EXPECTED_LAST_DIM[row["fusion"]]


def test_gmtm_full_and_text_only():
    full = run_one("bert", ("text", "audio", "visual"), batch_size=2, seed=0)
    text = run_one("bert", ("text",), batch_size=2, seed=0)
    assert full["output"].shape == (2, 1)
    assert text["output"].shape == (2, 1)
    assert full["finite"] and text["finite"]
    # Same architecture, so the parameter count must match.
    assert full["n_params"] == text["n_params"]
    assert full["n_params"] > 100_000


def test_metrics_match_hand_oracle():
    repo = metrics_from_repo(Y, YHAT)
    hand = metrics_by_hand(Y, YHAT)
    for key in repo:
        assert repo[key] == pytest.approx(hand[key], abs=1e-9)
    assert repo["MAE"] == pytest.approx(0.26)
    assert repo["Acc2"] == pytest.approx(1.0)
    assert repo["F1"] == pytest.approx(1.0)


def test_split_uniform_edges():
    from train_and_test import split_uniform_5, split_uniform_7

    # -3 is the left edge (bin 1). +3 is clipped into the last bin.
    assert split_uniform_7(np.array([-3.0]))[0] == 1
    assert split_uniform_7(np.array([3.0]))[0] == 7
    assert split_uniform_5(np.array([-3.0]))[0] == 1
    assert split_uniform_5(np.array([3.0]))[0] == 5


def test_committed_result_tables():
    expected_best_mae = {
        "main_results.csv": "TransformerLate",
        "ablation_results.csv": "text+audio+visual",
        "glove_results.csv": "LowRankTensorFusion",
        "ablation_glove_results.csv": "text+audio+visual",
        "mosi_bert_results.csv": "TransformerLate",
        "ablation_mosi_results.csv": "text+visual",
        "mosi_glove_results.csv": "GatedMultiTransfomer",
        "ablation_mosi_glove_results.csv": "text+audio+visual",
    }
    for title, path in TABLES:
        df = load_table(path)
        assert len(df) >= 6
        assert {"Fusion Method", "MAE", "Corr", "F1"}.issubset(df.columns)
        best = df.loc[df["MAE"].idxmin(), "Fusion Method"]
        assert best == expected_best_mae[path.name], (title, best, path.name)


def test_docs_are_substantial():
    required = [
        "architecture.md",
        "datasets.md",
        "training.md",
        "evaluation.md",
        "results.md",
        "reproducing.md",
        "repository_map.md",
    ]
    for name in required:
        text = (DOCS / name).read_text(encoding="utf-8")
        assert len(text) > 2000, f"{name} is too short ({len(text)} chars)"
        assert text.lstrip().startswith("#"), f"{name} should start with a heading"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Gated Multi-Transformer" in readme or "GMTM" in readme
    assert "examples/run_all_examples.py" in readme
