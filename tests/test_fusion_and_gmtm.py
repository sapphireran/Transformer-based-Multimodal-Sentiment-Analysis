"""CPU smoke tests against the real modules in model/models.py."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.fusion_forward_pass import run_fusions
from examples.gmtm_forward_pass import forward_once
from examples.lib.synthetic import BERT_SPEC, GLOVE_SPEC
from examples.toy_train_loop import main as toy_main


def test_fusion_shapes_bert():
    shapes = run_fusions(BERT_SPEC, batch_size=2, seed=0, device=torch.device("cpu"))
    by_name = dict(shapes)
    assert by_name["ConcatEarly [B,T,F]"] == (2, 50, BERT_SPEC.total_early)
    assert by_name["ConcatLate (already pooled)"] == (
        2,
        BERT_SPEC.visual + BERT_SPEC.audio + BERT_SPEC.text,
    )
    assert by_name["LowRankTensorFusion rank=8"] == (2, 32)
    assert by_name["EarlyFusionTransformer last-step"][0] == 2
    assert by_name["LateFusionTransformer last-step"] == (2, 32)


def test_fusion_shapes_glove():
    shapes = run_fusions(GLOVE_SPEC, batch_size=2, seed=0, device=torch.device("cpu"))
    by_name = dict(shapes)
    assert by_name["ConcatEarly [B,T,F]"] == (2, 50, GLOVE_SPEC.total_early)


def test_gmtm_full_and_text_only_shapes():
    full = forward_once("bert", ["text", "audio", "visual"], batch_size=2, seed=3, device=torch.device("cpu"))
    text = forward_once("bert", ["text"], batch_size=2, seed=3, device=torch.device("cpu"))
    assert tuple(full.shape) == (2, 1)
    assert tuple(text.shape) == (2, 1)
    # Different masks should change the output of a randomly-initialized GMTM.
    assert not torch.allclose(full, text)


def test_gmtm_glove_width():
    pred = forward_once("glove", ["audio", "visual"], batch_size=1, seed=4, device=torch.device("cpu"))
    assert tuple(pred.shape) == (1, 1)


def test_toy_train_runs(capsys):
    code = toy_main(["--steps", "3", "--batch-size", "8", "--seed", "1"])
    assert code == 0
    log = capsys.readouterr().out
    assert "train_L1=" in log
    assert "held-out" in log
