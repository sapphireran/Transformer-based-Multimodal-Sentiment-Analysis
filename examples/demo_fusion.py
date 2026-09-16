"""Forward-pass every fusion module on a synthetic BERT-shaped batch.

This is a shape-and-finiteness check, not a quality ranking. Parameter
counts are printed so the TFN 128k-wide vector is visible next to LMF.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from common import (
    AUDIO_DIM,
    TEXT_BERT_DIM,
    VISUAL_DIM,
    count_params,
    device,
    finite,
)
from synthetic_multimodal import random_batch

import models as M


def _last_hidden_lstm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def build_fusion_zoo(dev: torch.device) -> Dict[str, Tuple[nn.Module, Callable]]:
    """Return {name: (module_or_bundle, forward_fn)} for the main sweep."""

    zoo: Dict[str, Tuple[nn.Module, Callable]] = {}

    concat_early = M.ConcatEarly().to(dev)
    early_head = nn.Sequential(
        M.LSTM(VISUAL_DIM + AUDIO_DIM + TEXT_BERT_DIM, 64, dropout=False),
        M.MLP(64, 64, 1),
    ).to(dev)

    def run_concat_early(v, a, t):
        fused = concat_early([v, a, t])
        return early_head(fused)

    zoo["ConcatEarly"] = (nn.ModuleList([concat_early, early_head]), run_concat_early)

    late_enc = nn.ModuleList(
        [
            M.LSTM(VISUAL_DIM, 16, dropout=False).to(dev),
            M.LSTM(AUDIO_DIM, 24, dropout=False).to(dev),
            M.LSTM(TEXT_BERT_DIM, 32, dropout=False).to(dev),
        ]
    )
    late_fuse = M.ConcatLate().to(dev)
    late_head = M.MLP(16 + 24 + 32, 32, 1).to(dev)

    def run_concat_late(v, a, t):
        reps = [_last_hidden_lstm(enc, x) for enc, x in zip(late_enc, (v, a, t))]
        return late_head(late_fuse(reps))

    zoo["ConcatLate"] = (nn.ModuleList([late_enc, late_fuse, late_head]), run_concat_late)

    # Reduced widths, same recipe as train_main_bert.py (19 / 39 / 159).
    tfn_enc = nn.ModuleList(
        [
            M.GRUWithLinear(VISUAL_DIM, 16, 8, dropout=False, batch_first=True).to(dev),
            M.GRUWithLinear(AUDIO_DIM, 24, 12, dropout=False, batch_first=True).to(dev),
            M.GRUWithLinear(TEXT_BERT_DIM, 32, 16, dropout=False, batch_first=True).to(dev),
        ]
    )
    tfn = M.TensorFusion().to(dev)
    # GRUWithLinear without has_padding returns the full sequence from gru[0]
    # then a linear — we take the last step to keep the outer product small.
    tfn_head = M.MLP((8 + 1) * (12 + 1) * (16 + 1), 32, 1).to(dev)

    def _pool_gru(enc, x):
        out = enc(x)
        if out.dim() == 3:
            out = out[:, -1, :]
        return out

    def run_tfn(v, a, t):
        reps = [_pool_gru(enc, x) for enc, x in zip(tfn_enc, (v, a, t))]
        fused = tfn(reps)
        return tfn_head(fused)

    zoo["TensorFusion"] = (nn.ModuleList([tfn_enc, tfn, tfn_head]), run_tfn)

    lmf_enc = nn.ModuleList(
        [
            M.GRUWithLinear(VISUAL_DIM, 16, 8, dropout=False, batch_first=True).to(dev),
            M.GRUWithLinear(AUDIO_DIM, 24, 16, dropout=False, batch_first=True).to(dev),
            M.GRUWithLinear(TEXT_BERT_DIM, 32, 24, dropout=False, batch_first=True).to(dev),
        ]
    )
    lmf = M.LowRankTensorFusion([8, 16, 24], 16, rank=4).to(dev)
    lmf_head = M.MLP(16, 16, 1).to(dev)

    def run_lmf(v, a, t):
        reps = [_pool_gru(enc, x) for enc, x in zip(lmf_enc, (v, a, t))]
        return lmf_head(lmf(reps))

    zoo["LowRankTensorFusion"] = (nn.ModuleList([lmf_enc, lmf, lmf_head]), run_lmf)

    early_tr = M.EarlyFusionTransformer(n_features=VISUAL_DIM + AUDIO_DIM + TEXT_BERT_DIM).to(dev)
    early_tr_head = M.MLP(32, 32, 1).to(dev)

    def run_early_tr(v, a, t):
        return early_tr_head(early_tr([v, a, t]))

    zoo["TransformerEarly"] = (nn.ModuleList([early_tr, early_tr_head]), run_early_tr)

    late_tr_enc = nn.ModuleList(
        [
            M.TransformerSeq(VISUAL_DIM, 16).to(dev),
            M.TransformerSeq(AUDIO_DIM, 24).to(dev),
            M.TransformerSeq(TEXT_BERT_DIM, 32).to(dev),
        ]
    )
    late_tr = M.LateFusionTransformer(in_dim=16 + 24 + 32, embed_dim=16).to(dev)
    late_tr_head = M.MLP(16, 16, 1).to(dev)

    def run_late_tr(v, a, t):
        seqs = [enc(x) for enc, x in zip(late_tr_enc, (v, a, t))]
        return late_tr_head(late_tr(seqs))

    zoo["TransformerLate"] = (nn.ModuleList([late_tr_enc, late_tr, late_tr_head]), run_late_tr)
    return zoo


def run(verbose: bool = True) -> List[dict]:
    dev = device()
    torch.manual_seed(0)
    # Shorter T than 50 keeps the CPU demo snappy; widths stay MOSI-real.
    vision, audio, text = random_batch(batch_size=3, seq_len=8, text_dim=TEXT_BERT_DIM, device=dev)
    rows = []
    for name, (bundle, forward) in build_fusion_zoo(dev).items():
        bundle.eval()
        with torch.no_grad():
            out = forward(vision, audio, text)
        if out.dim() == 1:
            out = out.view(-1, 1)
        ok = finite(out) and out.shape[0] == vision.shape[0]
        row = {
            "name": name,
            "out_shape": tuple(out.shape),
            "params": count_params(bundle),
            "mean": float(out.mean().cpu()),
            "ok": ok,
        }
        rows.append(row)
        if verbose:
            status = "ok" if ok else "FAIL"
            print(
                f"[{status}] {name:22s}  out={row['out_shape']}  "
                f"params={row['params']:,}  mean={row['mean']:+.4f}"
            )
    return rows


def main() -> int:
    print(f"device={device()}")
    rows = run(verbose=True)
    failed = [r["name"] for r in rows if not r["ok"]]
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    print(f"fusion demos passed ({len(rows)} methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
