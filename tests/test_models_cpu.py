import torch

from examples.ablation_zeroing import score_subsets
from examples.attention_pooling import pool_and_weights
from examples.encoder_stack import run_encoders
from examples.forward_gmtm import run_forward
from examples.fusion_shapes import run_all_fusions
from examples.positional_embeddings import table
from examples.tiny_train_loop import train_toy


def test_gmtm_output_rank_and_finite():
    model, out, batch = run_forward(batch_size=3, seq_len=7)
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()
    assert model.n_modalities == 3


def test_fusion_modules_all_finite():
    rows = run_all_fusions(batch_size=2, seq_len=6)
    assert len(rows) == 7
    names = {r["name"] for r in rows}
    assert "ConcatEarly" in names
    assert "LowRankTensorFusion" in names
    assert all(r["finite"] for r in rows)


def test_encoders_run():
    rows = run_encoders(seq_len=6, batch_size=2)
    shapes = {r["name"]: r["shape"] for r in rows}
    assert shapes["TransformerSeq(vision → 16)"] == "2x6x16"
    assert shapes["Transformer last-step(audio)"].endswith("x16")
    assert shapes["MLP(mean text → 1)"] == "2x1"


def test_attention_weights_sum_to_one():
    x = torch.randn(3, 5, 8)
    out, weights = pool_and_weights(x)
    assert out.shape == (3, 8)
    assert torch.allclose(weights.sum(1), torch.ones(3), atol=1e-5)


def test_positional_table_detached():
    pos = table(seq_len=5, dim=8)
    assert pos.shape == (1, 5, 8)
    assert not pos.requires_grad


def test_ablation_flags():
    rows = {r["modalities"]: r for r in score_subsets(seed=1)}
    assert rows["text"]["nonzero_text"] and not rows["text"]["nonzero_audio"]
    assert rows["audio+visual"]["nonzero_audio"] and not rows["audio+visual"]["nonzero_text"]
    assert rows["text+audio+visual"]["nonzero_vision"]


def test_tiny_train_records_two_epochs():
    history = train_toy(steps_per_epoch=2, epochs=2, batch_size=4, seq_len=6)
    assert len(history["train"]) == 2
    assert len(history["valid"]) == 2
    assert all(v == v for v in history["train"] + history["valid"])
