#!/usr/bin/env python3
"""Parameter counts for GMTM and the BERT bake-off encoder/fusion/head stacks.

Mirrors the pairing in `train_main_bert.py` and the GMTM HParams in
`train_GMTM_bert.py` / `train_GMTM_glove.py`, but builds modules on CPU and
never loads a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.paths import ensure_model_on_path
from examples.lib.synthetic import BERT_SPEC, GLOVE_SPEC

ensure_model_on_path()
from models import (  # noqa: E402
    ConcatEarly,
    ConcatLate,
    EarlyFusionTransformer,
    GatedMultiTransfomerModel,
    GRUWithLinear,
    Identity,
    LateFusionTransformer,
    LowRankTensorFusion,
    LSTM,
    MLP,
    TensorFusion,
    TransformerSeq,
)


class RecordedGMTMHparams:
    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 64
    attn_mask = True
    output_dim = 1
    all_steps = False


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _stack_total(parts: Iterable[nn.Module]) -> int:
    return sum(n_params(m) for m in parts)


def bert_bakeoff() -> List[Tuple[str, int, str]]:
    v, a, t = BERT_SPEC.dims
    rows = []

    early = [Identity(), Identity(), Identity(), ConcatEarly(),
             nn.Sequential(LSTM(v + a + t, 1024, dropout=True, has_padding=True), MLP(1024, 1024, 1))]
    rows.append(("BERT ConcatEarly", _stack_total(early), "Identity×3 + concat + LSTM+MLP"))

    late = [
        LSTM(v, 64, dropout=True, has_padding=True),
        LSTM(a, 256, dropout=True, has_padding=True),
        LSTM(t, 1024, dropout=True, has_padding=True),
        ConcatLate(),
        MLP(1344, 1344, 1),
    ]
    rows.append(("BERT ConcatLate", _stack_total(late), "per-mod LSTM + MLP(1344)"))

    lrt = [
        GRUWithLinear(v, 64, 32, dropout=True, has_padding=True),
        GRUWithLinear(a, 256, 64, dropout=True, has_padding=True),
        GRUWithLinear(t, 1024, 256, dropout=True, has_padding=True),
        LowRankTensorFusion([32, 64, 256], 256, 32),
        MLP(256, 256, 1),
    ]
    rows.append(("BERT LowRankTensorFusion", _stack_total(lrt), "GRU+Linear + rank-32 LRTF"))

    tf = [
        GRUWithLinear(v, 64, 19, dropout=True, has_padding=True),
        GRUWithLinear(a, 256, 39, dropout=True, has_padding=True),
        GRUWithLinear(t, 1024, 159, dropout=True, has_padding=True),
        TensorFusion(),
        MLP(128000, 2048, 1),
    ]
    rows.append(("BERT TensorFusion", _stack_total(tf), "outer product → MLP(128000)"))

    te = [Identity(), Identity(), Identity(), EarlyFusionTransformer(n_features=v + a + t), MLP(64, 64, 1)]
    rows.append(("BERT TransformerEarly", _stack_total(te), "conv+4-layer encoder; head width 64"))

    tl = [
        TransformerSeq(v, 64),
        TransformerSeq(a, 128),
        TransformerSeq(t, 1024),
        LateFusionTransformer(in_dim=1216),
        MLP(32, 32, 1),
    ]
    rows.append(("BERT TransformerLate", _stack_total(tl), "3×TransformerSeq + late encoder"))
    return rows


def gmtm_rows() -> List[Tuple[str, int, str]]:
    bert = GatedMultiTransfomerModel(3, BERT_SPEC.dims, hyp_params=RecordedGMTMHparams)
    glove = GatedMultiTransfomerModel(3, GLOVE_SPEC.dims, hyp_params=RecordedGMTMHparams)
    return [
        ("GMTM BERT  [35,74,768] 4×64", n_params(bert), "Identity encoders/head in the script"),
        ("GMTM GloVe [35,74,300] 4×64", n_params(glove), "same HParams, thinner text proj"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print(f"{'stack':40s} {'params':>12s}  notes")
    print("-" * 90)
    for name, count, notes in gmtm_rows() + bert_bakeoff():
        print(f"{name:40s} {count:12,d}  {notes}")
    print()
    print("TensorFusion's head dominates the BERT bake-off (128000-d input).")
    print("GMTM's cost is the 3×3 cross-modal encoder grid, not the classifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
