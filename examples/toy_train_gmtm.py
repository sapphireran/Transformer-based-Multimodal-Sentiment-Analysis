#!/usr/bin/env python3
"""Overfit GMTM on a synthetic linear mixture of three modalities.

Proves the gated cross-modal stack can learn when the target is actually
visible in the features. No MOSI / MOSEI pickle, no CUDA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.metrics import classification_scores, regression_scores
from examples.lib.model_import import load_models
from examples.lib.synthetic import ToyMixtureSpec, make_toy_mixture


class ToyHParams:
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


N_FEATURES = (8, 12, 16)


def train(steps: int, batch_size: int, seq_len: int, seed: int) -> dict:
    models = load_models()
    torch.manual_seed(seed)
    streams, target = make_toy_mixture(
        batch_size=batch_size,
        seq_len=seq_len,
        n_features=N_FEATURES,
        spec=ToyMixtureSpec(),
        seed=seed,
    )
    model = models.GatedMultiTransfomerModel(3, list(N_FEATURES), hyp_params=ToyHParams)
    opt = torch.optim.Adam(model.parameters(), lr=8e-3)
    loss_fn = torch.nn.L1Loss()

    history = []
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = model(streams)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if step == 1 or step == steps or step % max(steps // 5, 1) == 0:
            print(f"step {step:3d}/{steps}  train MAE {history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final = model(streams).squeeze(-1).cpu().numpy()
    gold = target.squeeze(-1).cpu().numpy()
    reg = regression_scores(gold, final)
    clf = classification_scores(gold, final)
    report = {
        "first_mae": history[0],
        "last_mae": history[-1],
        "history": history,
        **reg,
        **clf,
    }
    print()
    print(
        "final  MAE={MAE:.4f}  Corr={Corr:.4f}  Acc7={Acc7_uniform:.3f}  Acc2={Acc2:.3f}".format(
            **report
        )
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.15,
        help="Require last MAE + this <= first MAE (sanity check)",
    )
    args = parser.parse_args(argv)

    report = train(args.steps, args.batch_size, args.seq_len, args.seed)
    dropped = report["first_mae"] - report["last_mae"]
    if dropped < args.min_drop:
        raise SystemExit(
            f"MAE only dropped {dropped:.4f} (first={report['first_mae']:.4f}, "
            f"last={report['last_mae']:.4f}); expected at least {args.min_drop}"
        )
    print(f"MAE dropped {dropped:.4f} (>= {args.min_drop}); toy train looks healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
