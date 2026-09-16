#!/usr/bin/env python3
"""Parameter counts for the fusion families at BERT and GloVe widths.

Constructs the same encoder / fusion / head objects as ``train_main_*.py``
and ``train_GMTM_*.py`` on CPU. Tensor Fusion's MLP on a 128000-d product
is counted from shapes without allocating that matrix (it is ~1 GiB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from gmtm_forward import PaperHParams, TinyHParams, count_parameters  # noqa: E402
from toy_data import AUDIO_DIM, BERT_DIM, GLOVE_DIM, VISION_DIM  # noqa: E402

add_model_to_path()


def _mlp_numel(indim: int, hiddim: int, outdim: int = 1) -> int:
    return indim * hiddim + hiddim + hiddim * outdim + outdim


def describe_zoo(text_dim: int) -> List[Dict]:
    import torch
    from models import (
        ConcatEarly,
        ConcatLate,
        EarlyFusionTransformer,
        GatedMultiTransfomerModel,
        Identity,
        LateFusionTransformer,
        LowRankTensorFusion,
        LSTM,
        MLP,
        TransformerSeq,
    )

    rows: List[Dict] = []
    total_in = VISION_DIM + AUDIO_DIM + text_dim
    backend = "bert" if text_dim == BERT_DIM else "glove"

    def add(name: str, n_params: int, note: str = "") -> None:
        rows.append({"backend": backend, "name": name, "parameters": int(n_params), "note": note})

    if text_dim == BERT_DIM:
        early_hidden = 1024
        late_hiddens = (64, 256, 1024)
        lmf_out = (32, 64, 256)
        lmf_fuse = 256
        late_tr = (64, 128, 1024)
        late_in = 1216  # 64+128+1024, matches train_main_bert.py
    else:
        early_hidden = 512
        late_hiddens = (64, 256, 512)
        lmf_out = (32, 64, 128)
        lmf_fuse = 128
        late_tr = (64, 128, 512)
        # train_main_glove.py passes in_dim=1792 even though 64+128+512 = 704.
        late_in = 1792

    # ConcatEarly: 3×Identity + LSTM(total_in → hidden) + MLP(hidden)
    lstm_early = LSTM(total_in, early_hidden, dropout=True, has_padding=True)
    mlp_early = MLP(early_hidden, early_hidden, 1)
    add(
        "ConcatEarly",
        count_parameters(ConcatEarly()) + count_parameters(lstm_early) + count_parameters(mlp_early),
        f"LSTM({total_in}→{early_hidden})+MLP",
    )

    # ConcatLate
    encoders = [
        LSTM(VISION_DIM, late_hiddens[0], dropout=True, has_padding=True),
        LSTM(AUDIO_DIM, late_hiddens[1], dropout=True, has_padding=True),
        LSTM(text_dim, late_hiddens[2], dropout=True, has_padding=True),
    ]
    late_concat_dim = sum(late_hiddens)
    add(
        "ConcatLate",
        sum(count_parameters(m) for m in encoders) + count_parameters(ConcatLate()) + count_parameters(MLP(late_concat_dim, late_concat_dim, 1)),
        f"LSTMs {late_hiddens} + MLP({late_concat_dim})",
    )

    lmf = LowRankTensorFusion(list(lmf_out), lmf_fuse, 32)
    add("LowRankTensorFusion", count_parameters(lmf) + count_parameters(MLP(lmf_fuse, lmf_fuse, 1)), f"rank 32, out {lmf_fuse} (encoders extra)")

    if text_dim == BERT_DIM:
        tfn_head = _mlp_numel(128000, 2048, 1)
        add("TensorFusion+head", tfn_head, "fusion has 0 params; MLP(128000,2048,1) counted from shapes")
    else:
        tfn_head = _mlp_numel(64000, 2048, 1)
        add("TensorFusion+head", tfn_head, "fusion has 0 params; MLP(64000,2048,1) counted from shapes")

    early_tr = EarlyFusionTransformer(n_features=total_in)
    add("EarlyFusionTransformer", count_parameters(early_tr) + count_parameters(Identity()), f"n_features={total_in}")

    enc_tr = [
        TransformerSeq(VISION_DIM, late_tr[0]),
        TransformerSeq(AUDIO_DIM, late_tr[1]),
        TransformerSeq(text_dim, late_tr[2]),
    ]
    late_module = LateFusionTransformer(in_dim=late_in, embed_dim=32)
    add(
        "LateFusionTransformer",
        sum(count_parameters(m) for m in enc_tr) + count_parameters(late_module) + count_parameters(MLP(32, 32, 1)),
        f"TransformerSeq {late_tr}, fusion in_dim={late_in}"
        + ("" if text_dim == BERT_DIM else "; script in_dim=1792 vs encoder sum 704"),
    )

    gmtm_paper = GatedMultiTransfomerModel(3, [VISION_DIM, AUDIO_DIM, text_dim], hyp_params=PaperHParams)
    gmtm_tiny = GatedMultiTransfomerModel(3, [VISION_DIM, AUDIO_DIM, text_dim], hyp_params=TinyHParams)
    add("GMTM paper HParams", count_parameters(gmtm_paper), "embed 64, 4 layers, 4 heads")
    add("GMTM TinyHParams", count_parameters(gmtm_tiny), "embed 16, 1 layer, 2 heads (examples)")

    del lstm_early, mlp_early, encoders, lmf, early_tr, enc_tr, late_module, gmtm_paper, gmtm_tiny
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bert", "glove", "both"), default="both")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    backends = []
    if args.backend in ("bert", "both"):
        backends.append(BERT_DIM)
    if args.backend in ("glove", "both"):
        backends.append(GLOVE_DIM)

    all_rows: List[Dict] = []
    for text_dim in backends:
        rows = describe_zoo(text_dim)
        all_rows.extend(rows)
        label = "BERT" if text_dim == BERT_DIM else "GloVe"
        print(f"== {label} (text_dim={text_dim}) ==")
        width = max(len(row["name"]) for row in rows)
        for row in rows:
            print(f"  {row['name']:<{width}}  {row['parameters']:>12,}  {row['note']}")
        print()

    if args.write_json:
        out = ensure_output_dir() / "count_params.json"
        out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
