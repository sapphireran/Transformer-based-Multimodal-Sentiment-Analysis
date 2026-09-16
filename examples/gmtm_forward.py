#!/usr/bin/env python3
"""Run GatedMultiTransfomerModel on a synthetic batch and dump activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from toy_data import AUDIO_DIM, BERT_DIM, GLOVE_DIM, VISION_DIM, make_toy_batch  # noqa: E402

add_model_to_path()


class TinyHParams:
    """CPU-friendly GMTM settings; heads must divide embed_dim."""

    num_heads = 2
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


class PaperHParams:
    """Hyperparameters from ``train_GMTM_bert.py`` (slow on CPU)."""

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


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bert", "glove"), default="bert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--preset", choices=("tiny", "paper"), default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from models import GatedMultiTransfomerModel

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    text_dim = BERT_DIM if args.backend == "bert" else GLOVE_DIM
    hyp = TinyHParams if args.preset == "tiny" else PaperHParams
    batch = make_toy_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_backend=args.backend,
        seed=11,
    )

    model = GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[VISION_DIM, AUDIO_DIM, text_dim],
        hyp_params=hyp,
    ).to(device)
    model.eval()

    vision = torch.from_numpy(batch.vision).to(device)
    audio = torch.from_numpy(batch.audio).to(device)
    text = torch.from_numpy(batch.text).to(device)

    with torch.no_grad():
        pred = model([vision, audio, text])

    n_params = count_parameters(model)
    print(f"preset={args.preset}  backend={args.backend}  device={device}")
    print(f"parameters={n_params:,}")
    print(f"input   vision{tuple(vision.shape)} audio{tuple(audio.shape)} text{tuple(text.shape)}")
    print(f"output  {tuple(pred.shape)}  min={pred.min().item():.4f}  max={pred.max().item():.4f}")
    print("predictions:")
    for i, (y_hat, y) in enumerate(zip(pred.view(-1).tolist(), batch.labels.reshape(-1).tolist())):
        print(f"  clip {i:02d}  pred={y_hat:+.4f}  toy_label={y:+.4f}")

    if not torch.isfinite(pred).all():
        print("error: non-finite GMTM output")
        return 1

    if args.write_json:
        payload = {
            "preset": args.preset,
            "backend": args.backend,
            "parameters": n_params,
            "output_shape": list(pred.shape),
            "predictions": pred.view(-1).cpu().tolist(),
            "labels": batch.labels.reshape(-1).tolist(),
        }
        out = ensure_output_dir() / "gmtm_forward.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
