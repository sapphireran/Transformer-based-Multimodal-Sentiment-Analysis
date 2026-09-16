import torch

from examples.paths import ensure_model_on_path
from examples.run_fusion_forward import run as run_fusion
from examples.synthetic_affect import AffectShapes, batch_from_split, make_split_arrays

ensure_model_on_path()
from models import ConcatEarly, ConcatLate, GatedMultiTransfomerModel, TensorFusion  # noqa: E402


def test_concat_and_tfn_shapes_direct():
    concat_early = ConcatEarly()
    x = [
        torch.zeros(2, 5, 35),
        torch.zeros(2, 5, 74),
        torch.zeros(2, 5, 768),
    ]
    assert concat_early(x).shape == (2, 5, 877)

    concat_late = ConcatLate()
    assert concat_late([torch.zeros(2, 64), torch.zeros(2, 256), torch.zeros(2, 1024)]).shape == (2, 1344)

    tfn = TensorFusion()
    out = tfn([torch.zeros(2, 19), torch.zeros(2, 39), torch.zeros(2, 159)])
    assert out.shape == (2, 128000)


def test_gmtm_default_forward_shape():
    shapes = AffectShapes.for_embedding("bert", max_len=8)
    split = make_split_arrays(2, shapes=shapes, seed=0)
    vision, audio, text, _ = batch_from_split(split, 2)
    model = GatedMultiTransfomerModel(3, [shapes.visual, shapes.audio, shapes.text])
    model.eval()
    with torch.no_grad():
        y = model([vision, audio, text])
    assert y.shape == (2, 1)
    assert torch.isfinite(y).all()


def test_fusion_runner_contract():
    report = run_fusion(batch_size=2, seq_len=8, seed=0)
    assert report["ok"], report.get("mismatches")
    assert report["modules"]["ConcatEarly"]["output_shape"] == [2, 8, 877]
    assert report["modules"]["TensorFusion"]["output_shape"] == [2, 128000]
    assert report["modules"]["GMTM_default_hparams"]["output_shape"] == [2, 1]
