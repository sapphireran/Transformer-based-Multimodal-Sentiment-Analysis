"""Print input / output ranks for every fusion module the scripts use."""

from __future__ import annotations

from typing import Callable

import torch

from examples.common import BERT_LAYOUT, cpu_device, finite, format_shape
from examples.synthetic_data import make_batch


def _cat_early(vision, audio, text):
    from models import ConcatEarly

    return ConcatEarly()([vision, audio, text])


def _cat_late(vision, audio, text):
    from models import ConcatLate

    # Late fusion in the scripts sees pooled [B, F] vectors.
    pooled = [vision.mean(1), audio.mean(1), text.mean(1)]
    return ConcatLate()(pooled)


def _tfn(vision, audio, text):
    from models import TensorFusion

    pooled = [vision.mean(1)[:, :8], audio.mean(1)[:, :8], text.mean(1)[:, :8]]
    return TensorFusion()(pooled)


def _lrtf(vision, audio, text):
    from models import LowRankTensorFusion

    vis = vision.mean(1)[:, :16]
    aud = audio.mean(1)[:, :16]
    txt = text.mean(1)[:, :16]
    fusion = LowRankTensorFusion([16, 16, 16], output_dim=8, rank=4)
    return fusion([vis, aud, txt])


def _transformer_fusion(vision, audio, text):
    from models import TransformerFusion

    d = 16
    # Shared width: the first d channels of each mean-pooled modality.
    mods = [vision.mean(1)[:, :d], audio.mean(1)[:, :d], text.mean(1)[:, :d]]
    return TransformerFusion(d_model=d, nhead=4, num_layers=1, dropout=0.0)(mods)


def _early_transformer(vision, audio, text):
    from models import EarlyFusionTransformer

    return EarlyFusionTransformer(n_features=BERT_LAYOUT.concat_dim)([vision, audio, text])


def _late_transformer(vision, audio, text):
    from models import LateFusionTransformer, TransformerSeq

    enc_v = TransformerSeq(BERT_LAYOUT.visual, 16)
    enc_a = TransformerSeq(BERT_LAYOUT.audio, 16)
    enc_t = TransformerSeq(BERT_LAYOUT.text, 16)
    seqs = [enc_v(vision), enc_a(audio), enc_t(text)]
    return LateFusionTransformer(in_dim=48, embed_dim=16)(seqs)


CASES: list[tuple[str, Callable]] = [
    ("ConcatEarly", _cat_early),
    ("ConcatLate (mean-pooled)", _cat_late),
    ("TensorFusion (8-D slice)", _tfn),
    ("LowRankTensorFusion", _lrtf),
    ("TransformerFusion", _transformer_fusion),
    ("EarlyFusionTransformer", _early_transformer),
    ("LateFusionTransformer", _late_transformer),
]


def run_all_fusions(batch_size: int = 3, seq_len: int = 10) -> list[dict]:
    device = cpu_device()
    batch = make_batch(batch_size=batch_size, seq_len=seq_len)
    vision, audio, text = [t.to(device) for t in batch.as_list()]
    rows = []
    for name, fn in CASES:
        torch.manual_seed(0)
        out = fn(vision, audio, text)
        if not finite(out):
            raise RuntimeError(f"{name} produced non-finite values")
        rows.append(
            {
                "name": name,
                "out_shape": format_shape(out),
                "finite": True,
                "abs_mean": float(out.detach().abs().mean()),
            }
        )
    return rows


def main() -> int:
    print("Fusion module output ranks on a synthetic BERT batch")
    print(f"  inputs: B=3 T=10  F={BERT_LAYOUT.as_list}  concat={BERT_LAYOUT.concat_dim}")
    for row in run_all_fusions():
        print(f"  {row['name']:<28} -> {row['out_shape']:<16} |mean|={row['abs_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
