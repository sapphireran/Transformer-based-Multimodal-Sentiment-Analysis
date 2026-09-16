"""Overfit GatedMultiTransfomerModel on synthetic labels.

This is the CPU stand-in for train_GMTM_bert.py: same 3-slot module, same
zeroing protocol, L1 loss, AdamW. The batch is fake, the hidden size is
smaller than HParams.embed_dim=64, and we only take a handful of steps —
enough to prove the backward path and that text-only zeros do not crash.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import count_parameters, device, feature_dims
from synthetic_data import make_corpus


class ToyHParams:
    num_heads = 2
    layers = 2
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


def _build_gmtm(text: str = "bert"):
    from models import GatedMultiTransfomerModel

    dims = list(feature_dims(text))
    model = GatedMultiTransfomerModel(3, dims, hyp_params=ToyHParams)
    return model.to(device()), dims


def _step_loop(model, vision, audio, text, labels, steps: int, lr: float) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    losses = []
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = model([vision, audio, text])
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        loss = loss_fn(pred, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        opt.step()
        losses.append(float(loss.item()))
        print(f"step {step:02d}  L1={loss.item():.4f}  pred_mean={pred.mean().item():.3f}")
    return losses


def run_full_and_text_only(steps: int = 8) -> dict[str, list[float]]:
    torch.manual_seed(7)
    corpus = make_corpus(n=6, text="bert", min_len=8, max_len=12, seed=7)
    vision, audio, text, labels = corpus.padded(max_len=12)

    print(f"device: {device()}")
    model, dims = _build_gmtm("bert")
    print(f"GMTM n_features={dims}  embed_dim={ToyHParams.embed_dim}  params={count_parameters(model)}")
    print(f"batch vision{tuple(vision.shape)} audio{tuple(audio.shape)} text{tuple(text.shape)}")
    print()
    print("--- full trio ---")
    full_losses = _step_loop(model, vision, audio, text, labels, steps=steps, lr=3e-3)

    print()
    print("--- text only (visual/audio zeroed, same architecture) ---")
    model_txt, _ = _build_gmtm("bert")
    v0, a0, t0, y0 = corpus.zero_modalities({"text"}, max_len=12)
    text_losses = _step_loop(model_txt, v0, a0, t0, y0, steps=steps, lr=3e-3)

    print()
    print(f"full     first L1={full_losses[0]:.4f}  last L1={full_losses[-1]:.4f}")
    print(f"text-only first L1={text_losses[0]:.4f}  last L1={text_losses[-1]:.4f}")
    if full_losses[-1] >= full_losses[0]:
        print("note: last full-trio L1 did not drop; rerun or add steps (synthetic noise is small).")
    return {"full": full_losses, "text_only": text_losses}


def main() -> None:
    run_full_and_text_only(steps=8)


if __name__ == "__main__":
    main()
