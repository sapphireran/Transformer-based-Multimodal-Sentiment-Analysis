"""One CPU backward step on GMTM must stay finite."""

from __future__ import annotations

import torch
import torch.nn as nn

from models import GatedMultiTransfomerModel
from synthetic import make_synthetic_batch, modality_dims


class TinyHParams:
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


def test_single_gmtm_step_is_finite():
    torch.manual_seed(0)
    batch = make_synthetic_batch(batch_size=2, seq_len=6, seed=0)
    model = GatedMultiTransfomerModel(3, list(modality_dims("bert")), hyp_params=TinyHParams)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(batch.as_list())
    loss = nn.functional.l1_loss(out, batch.labels)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    with torch.no_grad():
        again = model(batch.as_list())
    assert torch.isfinite(again).all()
    assert again.shape == (2, 1)
