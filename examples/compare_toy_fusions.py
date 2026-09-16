#!/usr/bin/env python3
"""Fit a mean-pool MLP (late-concat analogue) vs tiny GMTM on the same toy split.

Not a MOSEI result. It shows that GMTM's extra graph is not required to fit
the planted text cue, and that both code paths share ``compute_sentiment_metrics``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from gmtm_forward import TinyHParams, count_parameters  # noqa: E402
from toy_data import AUDIO_DIM, GLOVE_DIM, VISION_DIM, iter_torch_batches, make_toy_split  # noqa: E402
from train_toy_gmtm import _epoch, _predict  # noqa: E402

add_model_to_path()

from metrics import compute_sentiment_metrics, format_metrics  # noqa: E402


class MeanPoolConcatRegressor:
    """Late-concat analogue: mean-pool each stream, MLP on the stack."""

    def __init__(self, text_dim: int, hidden: int = 64):
        import torch
        from torch import nn

        in_dim = VISION_DIM + AUDIO_DIM + text_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def to(self, device: str):
        self.net.to(device)
        return self

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def parameters(self):
        return self.net.parameters()

    def __call__(self, mods):
        import torch

        pooled = [m.mean(dim=1) for m in mods]
        return self.net(torch.cat(pooled, dim=1))


def _fit(model, splits, device: str, epochs: int, lr: float):
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, splits["train"], opt, device)
        val_pred, val_y = _predict(model, splits["valid"], device)
        val_scores = compute_sentiment_metrics(val_y, val_pred)
        history.append({"epoch": epoch, "train_mae": train_loss, **val_scores})
    test_pred, test_y = _predict(model, splits["test"], device)
    test_scores = compute_sentiment_metrics(test_y, test_pred)
    return history, test_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-valid", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from models import GatedMultiTransfomerModel

    torch.manual_seed(args.seed)
    splits = make_toy_split(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_backend="glove",
        seed=args.seed,
    )

    mlp = MeanPoolConcatRegressor(GLOVE_DIM).to(args.device)
    gmtm = GatedMultiTransfomerModel(
        3, [VISION_DIM, AUDIO_DIM, GLOVE_DIM], hyp_params=TinyHParams
    ).to(args.device)

    print(f"mean-pool MLP params={count_parameters(mlp.net):,}")
    print(f"tiny GMTM     params={count_parameters(gmtm):,}")
    print()

    results = {}
    for name, model in (("mean_pool_mlp", mlp), ("tiny_gmtm", gmtm)):
        history, test_scores = _fit(model, splits, args.device, args.epochs, args.lr)
        results[name] = {
            "history": history,
            "test": {k: float(v) for k, v in test_scores.items()},
        }
        last = history[-1]
        print(
            f"{name}: last val MAE={last['MAE']:.4f}  "
            f"test MAE={test_scores['MAE']:.4f}  test Corr={test_scores['Corr']:.3f}"
        )
        print(format_metrics(test_scores))
        print()

    mlp_mae = results["mean_pool_mlp"]["test"]["MAE"]
    gmtm_mae = results["tiny_gmtm"]["test"]["MAE"]
    # Both should beat a ~1.2 MAE constant-0 predictor on this planted cue.
    if mlp_mae > 1.0 and gmtm_mae > 1.0:
        print("error: neither model beat MAE 1.0 on the toy cue")
        return 1

    if args.write_json:
        out = ensure_output_dir() / "compare_toy_fusions.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
