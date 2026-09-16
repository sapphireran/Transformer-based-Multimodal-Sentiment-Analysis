#!/usr/bin/env python3
"""Forward-pass every fusion module on a synthetic batch and print shapes.

This is the CPU-friendly counterpart to ``train_main_bert.py``: it does not
load pickles or checkpoints, it only checks that each fusion family accepts
the documented tensor layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from toy_data import AUDIO_DIM, BERT_DIM, GLOVE_DIM, VISION_DIM, make_toy_batch  # noqa: E402

add_model_to_path()


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tensors(batch, device):
    import torch

    vision = torch.from_numpy(batch.vision).to(device)
    audio = torch.from_numpy(batch.audio).to(device)
    text = torch.from_numpy(batch.text).to(device)
    return vision, audio, text


def _run_named(name: str, fn: Callable, records: List[Dict]) -> None:
    import torch

    try:
        with torch.no_grad():
            out = fn()
        shape = tuple(out.shape)
        finite = bool(torch.isfinite(out).all().item())
        print(f"{name:<28}  shape={shape}  finite={finite}")
        records.append({"name": name, "shape": list(shape), "finite": finite, "ok": True})
    except Exception as exc:  # pragma: no cover - surfaced in the printed table
        print(f"{name:<28}  ERROR: {type(exc).__name__}: {exc}")
        records.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def build_and_run(text_backend: str, batch_size: int, seq_len: int, device_name: str) -> List[Dict]:
    import torch
    from models import (
        ConcatEarly,
        ConcatLate,
        EarlyFusionTransformer,
        LateFusionTransformer,
        LowRankTensorFusion,
        TensorFusion,
        TransformerFusion,
        TransformerSeq,
    )

    device = _device(device_name)
    text_dim = BERT_DIM if text_backend == "bert" else GLOVE_DIM
    batch = make_toy_batch(
        batch_size=batch_size, seq_len=seq_len, text_backend=text_backend, seed=3
    )
    vision, audio, text = _tensors(batch, device)
    records: List[Dict] = []

    print(f"device={device}  backend={text_backend}  B={batch_size} T={seq_len} text_dim={text_dim}")
    print()

    def concat_early():
        return ConcatEarly()( [vision, audio, text] )

    def concat_late():
        # Flatten time so ConcatLate sees [B, F] vectors, as the late-LSTM path does.
        mods = [vision.mean(dim=1), audio.mean(dim=1), text.mean(dim=1)]
        return ConcatLate()(mods)

    def tensor_fusion():
        mods = [vision.mean(dim=1)[:, :8], audio.mean(dim=1)[:, :8], text.mean(dim=1)[:, :8]]
        return TensorFusion()(mods)

    def low_rank():
        mods = [vision.mean(dim=1), audio.mean(dim=1), text.mean(dim=1)]
        fusion = LowRankTensorFusion(
            [VISION_DIM, AUDIO_DIM, text_dim], output_dim=32, rank=4
        ).to(device)
        return fusion(mods)

    def transformer_fusion():
        # Project each pooled modality to the same d_model before stacking.
        d_model = 32
        mods = []
        for tensor in (vision, audio, text):
            pooled = tensor.mean(dim=1)
            proj = torch.nn.Linear(pooled.shape[-1], d_model).to(device)
            mods.append(proj(pooled))
        return TransformerFusion(d_model=d_model, nhead=4, num_layers=1).to(device)(mods)

    def early_transformer():
        fusion = EarlyFusionTransformer(n_features=VISION_DIM + AUDIO_DIM + text_dim).to(device)
        return fusion([vision, audio, text])

    def late_transformer():
        enc_v = TransformerSeq(VISION_DIM, 32).to(device)
        enc_a = TransformerSeq(AUDIO_DIM, 32).to(device)
        enc_t = TransformerSeq(text_dim, 32).to(device)
        encoded = [enc_v(vision), enc_a(audio), enc_t(text)]
        fusion = LateFusionTransformer(in_dim=96, embed_dim=32).to(device)
        return fusion(encoded)

    _run_named("ConcatEarly", concat_early, records)
    _run_named("ConcatLate", concat_late, records)
    _run_named("TensorFusion (8-d slice)", tensor_fusion, records)
    _run_named("LowRankTensorFusion", low_rank, records)
    _run_named("TransformerFusion", transformer_fusion, records)
    _run_named("EarlyFusionTransformer", early_transformer, records)
    _run_named("LateFusionTransformer", late_transformer, records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bert", "glove"), default="bert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    records = build_and_run(args.backend, args.batch_size, args.seq_len, args.device)
    failed = [row for row in records if not row.get("ok")]
    if args.write_json:
        out = ensure_output_dir() / "fusion_zoo.json"
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
