"""Few-step L1 training on synthetic, text-correlated labels.

Uses the real ``GatedMultiTransfomerModel`` with tiny hyperparameters so
the example stays on CPU. Success criterion: train MAE drops versus the
first epoch, and the fitted model beats a constant-mean predictor on the
held-out synthetic test split.

This is *not* a MOSEI result. It only shows that the GMTM graph is
trainable end-to-end with the same loss the real scripts use.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import AUDIO_DIM, VISUAL_DIM, count_params, device, ensure_output_dir
from eval_protocol import format_metrics, regression_and_classification
from synthetic_multimodal import SyntheticConfig, make_dataloaders

import models as M


class TinyHParams:
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


TEXT_DIM = 32
SEQ_LEN = 12
EPOCHS = 8
BATCH = 8


def build_model(dev: torch.device) -> nn.Module:
    return M.GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[VISUAL_DIM, AUDIO_DIM, TEXT_DIM],
        hyp_params=TinyHParams,
    ).to(dev)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for vision, audio, text, y in loader:
        vision, audio, text = vision.to(dev), audio.to(dev), text.to(dev)
        out = model([vision, audio, text])
        if out.dim() == 1:
            out = out.view(-1, 1)
        preds.append(out.cpu())
        labels.append(y.cpu())
    return torch.cat(preds, 0), torch.cat(labels, 0)


def epoch_mae(model: nn.Module, loader: DataLoader, dev: torch.device, train: bool, opt=None) -> float:
    criterion = nn.L1Loss()
    model.train(train)
    total = 0.0
    count = 0
    for vision, audio, text, y in loader:
        vision, audio, text, y = vision.to(dev), audio.to(dev), text.to(dev), y.to(dev)
        if train:
            opt.zero_grad(set_to_none=True)
        out = model([vision, audio, text])
        if out.dim() == 1:
            out = out.view(-1, 1)
        loss = criterion(out, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 8.0)
            opt.step()
        total += float(loss.item()) * y.size(0)
        count += y.size(0)
    return total / max(count, 1)


def run(verbose: bool = True) -> dict:
    dev = device()
    torch.manual_seed(3)
    cfg = SyntheticConfig(
        n_train=80,
        n_valid=20,
        n_test=20,
        seq_len=SEQ_LEN,
        text_dim=TEXT_DIM,
        text_correlated=True,
        seed=3,
    )
    loaders = make_dataloaders(cfg, batch_size=BATCH)
    model = build_model(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    if verbose:
        print(
            f"device={dev}  params={count_params(model):,}  "
            f"train clips={cfg.n_train}  T={SEQ_LEN}  text_dim={TEXT_DIM}"
        )

    history = []
    for epoch in range(1, EPOCHS + 1):
        tr = epoch_mae(model, loaders["train"], dev, train=True, opt=opt)
        va = epoch_mae(model, loaders["valid"], dev, train=False)
        history.append((tr, va))
        if verbose:
            print(f"epoch {epoch:02d}  train L1={tr:.4f}  valid L1={va:.4f}")

    pred, gold = predict(model, loaders["test"], dev)
    metrics = regression_and_classification(gold, pred)
    mean_pred = torch.full_like(gold, gold.mean())
    baseline = regression_and_classification(gold, mean_pred)

    if verbose:
        print("test GMTM     ", format_metrics(metrics))
        print("test mean-base", format_metrics(baseline))

    out_dir = ensure_output_dir()
    log_path = out_dir / "toy_train_metrics.txt"
    log_path.write_text(
        "history=" + repr(history) + "\n"
        "gmtm=" + format_metrics(metrics) + "\n"
        "mean=" + format_metrics(baseline) + "\n",
        encoding="utf-8",
    )

    return {
        "history": history,
        "metrics": metrics,
        "baseline": baseline,
        "first_train": history[0][0],
        "last_train": history[-1][0],
    }


def main() -> int:
    result = run(verbose=True)
    dropped = result["last_train"] < result["first_train"]
    beat_mean = result["metrics"]["MAE"] < result["baseline"]["MAE"]
    if not dropped:
        print("train L1 did not drop — graph may not be training", file=sys.stderr)
        return 1
    if not beat_mean:
        print(
            "test MAE did not beat the constant-mean baseline "
            f"({result['metrics']['MAE']:.4f} vs {result['baseline']['MAE']:.4f})",
            file=sys.stderr,
        )
        return 1
    print("toy training demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
