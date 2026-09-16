"""Forward-pass ranks for every fusion in the CPU zoo."""

from __future__ import annotations

import torch

from msa_lab.fusion_zoo import build_fusion_zoo
from msa_lab.synthetic import make_sentiment_batch


def test_zoo_output_ranks():
    batch = make_sentiment_batch(batch_size=4, seq_len=10, preset="toy", seed=2)
    zoo = {spec.name: spec for spec in build_fusion_zoo(batch.dims())}

    early = zoo["ConcatEarly"].prepare(batch)
    assert early.shape == (4, 10, 8 + 12 + 16)

    late = zoo["ConcatLate"].prepare(batch)
    assert late.shape == (4, 8 + 12 + 16)

    tensor = zoo["TensorFusion"].prepare(batch)
    assert tensor.shape == (4, 9 * 13 * 17)

    lmf = zoo["LowRankTensorFusion"].prepare(batch)
    assert lmf.shape == (4, 16)

    trans = zoo["TransformerFusion"].prepare(batch)
    assert trans.shape == (4, 16)

    gated = zoo["GatedMultiTransformer"].prepare(batch)
    assert gated.shape == (4, 1)


def test_tensor_fusion_single_modality_passthrough():
    from models import TensorFusion

    fusion = TensorFusion()
    only = torch.randn(3, 5)
    out = fusion([only])
    assert torch.equal(out, only)


def test_concat_late_flattens_extra_dims():
    from models import ConcatLate

    fusion = ConcatLate()
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5)
    out = fusion([a, b])
    assert out.shape == (2, 12 + 5)
