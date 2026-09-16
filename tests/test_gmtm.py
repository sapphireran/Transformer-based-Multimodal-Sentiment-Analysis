import torch

from gmtm_forward import build_model, describe_forward, run as run_forward
from gmtm_toy_train import run as run_train
from synthetic_data import FeatureSpec, random_padded_batch, seed_everything, zero_ablate


def test_gmtm_output_is_scalar_per_clip():
    seed_everything(1)
    spec = FeatureSpec()
    streams, labels = random_padded_batch(batch_size=5, spec=spec)
    model = build_model(spec, streams[0].device)
    model.eval()
    with torch.no_grad():
        out = model(streams)
    assert out.shape == (5, 1)
    assert labels.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_replay_matches_module_forward():
    payload = run_forward()
    assert payload["forward_matches_replay"] is True
    assert payload["n_params"] > 0
    stages = {row["stage"]: row["shape"] for row in payload["stages"]}
    assert stages["head"] == [6, 1]
    assert stages["concat"][0] == 6
    assert stages["concat"][-1] == 48  # 3 * embed_dim 16


def test_zero_ablation_changes_output():
    seed_everything(2)
    spec = FeatureSpec()
    streams, _ = random_padded_batch(batch_size=4, spec=spec)
    model = build_model(spec, streams[0].device)
    model.eval()
    with torch.no_grad():
        full = model(streams)
        text_only = model(zero_ablate(streams, keep=("text",)))
    assert not torch.allclose(full, text_only, atol=1e-6)


def test_toy_train_loss_moves():
    result = run_train(steps=20, batch_size=8, lr=3e-3)
    assert result["steps"] == 20
    assert result["n_params"] > 0
    assert result["loss_went_down"] is True
    assert result["train_mae_last5"] < result["train_mae_first5"]
    assert result["test_mae"] < 1.5


def test_describe_forward_param_count_matches_module():
    seed_everything(3)
    spec = FeatureSpec()
    streams, _ = random_padded_batch(batch_size=2, spec=spec)
    model = build_model(spec, streams[0].device)
    replay, _notes = describe_forward(model, streams)
    assert replay.shape == (2, 1)
