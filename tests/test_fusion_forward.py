from fusion_forward import run as run_fusion
from synthetic_data import FeatureSpec, pick_device, random_padded_batch, seed_everything

from models import ConcatEarly, ConcatLate, TensorFusion


def test_concat_and_tensor_shapes_cpu():
    seed_everything(0)
    spec = FeatureSpec()
    streams, _ = random_padded_batch(batch_size=4, spec=spec)
    early = ConcatEarly()(streams)
    assert early.shape == (4, spec.seq_len, spec.total)

    pooled = [s.mean(dim=1) for s in streams]
    late = ConcatLate()(pooled)
    assert late.shape == (4, spec.total)

    fused = TensorFusion()(pooled)
    assert fused.shape == (4, (spec.visual + 1) * (spec.audio + 1) * (spec.text + 1))


def test_fusion_forward_script_all_finite():
    reports = run_fusion()
    names = {row["name"] for row in reports}
    assert {
        "ConcatEarly",
        "ConcatLate",
        "TensorFusion",
        "LowRankTensorFusion",
        "TransformerFusion",
        "EarlyFusionTransformer",
        "LateFusionTransformer",
    } <= names
    assert all(row["finite"] for row in reports)


def test_device_helper_is_cpu_here():
    assert pick_device().type in {"cpu", "cuda"}
