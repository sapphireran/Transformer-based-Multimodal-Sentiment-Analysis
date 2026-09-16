#!/usr/bin/env python3
"""Fit a tiny GMTM on synthetic multimodal sentiment and report metrics.

This is not a MOSEI result. It exists so the gated multi-transformer graph,
the L1 + AdamW loop, and ``compute_sentiment_metrics`` can be exercised
without the CMU pickles or a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from gmtm_forward import TinyHParams, count_parameters  # noqa: E402
from toy_data import AUDIO_DIM, BERT_DIM, GLOVE_DIM, VISION_DIM, iter_torch_batches, make_toy_split  # noqa: E402

add_model_to_path()

from metrics import compute_sentiment_metrics, format_metrics  # noqa: E402


def _predict(model, batches, device: str):
    import torch

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for vision, audio, text, y in iter_torch_batches(batches, device=device):
            out = model([vision, audio, text])
            preds.append(out.detach().cpu().reshape(-1))
            labels.append(y.detach().cpu().reshape(-1))
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def _epoch(model, batches, optimizer, device: str) -> float:
    import torch

    model.train()
    total = 0.0
    count = 0
    loss_fn = torch.nn.L1Loss()
    for vision, audio, text, y in iter_torch_batches(batches, device=device):
        optimizer.zero_grad(set_to_none=True)
        pred = model([vision, audio, text])
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        optimizer.step()
        total += float(loss.item()) * y.shape[0]
        count += y.shape[0]
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bert", "glove"), default="glove")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-valid", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from models import GatedMultiTransfomerModel

    torch.manual_seed(args.seed)
    device = args.device
    text_dim = BERT_DIM if args.backend == "bert" else GLOVE_DIM
    splits = make_toy_split(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_backend=args.backend,
        seed=args.seed,
    )

    model = GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[VISION_DIM, AUDIO_DIM, text_dim],
        hyp_params=TinyHParams,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(
        f"tiny GMTM  params={count_parameters(model):,}  "
        f"backend={args.backend}  device={device}"
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _epoch(model, splits["train"], optimizer, device)
        val_pred, val_y = _predict(model, splits["valid"], device)
        val_scores = compute_sentiment_metrics(val_y, val_pred)
        row = {"epoch": epoch, "train_mae": train_loss, **val_scores}
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_L1={train_loss:.4f}  "
            f"val_MAE={val_scores['MAE']:.4f}  val_Corr={val_scores['Corr']:.3f}  "
            f"val_Acc2={val_scores['Acc2']:.3f}"
        )

    test_pred, test_y = _predict(model, splits["test"], device)
    test_scores = compute_sentiment_metrics(test_y, test_pred)
    print("\n== test ==")
    print(format_metrics(test_scores))

    if args.write_json:
        payload = {
            "backend": args.backend,
            "epochs": args.epochs,
            "parameters": count_parameters(model),
            "history": history,
            "test": {k: float(v) for k, v in test_scores.items()},
        }
        out = ensure_output_dir() / "train_toy_gmtm.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    first = history[0]["MAE"]
    last = history[-1]["MAE"]
    if last > first + 0.05:
        print(f"warning: validation MAE rose ({first:.4f} → {last:.4f})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
