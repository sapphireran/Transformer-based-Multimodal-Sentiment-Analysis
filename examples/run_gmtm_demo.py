#!/usr/bin/env python3
"""Forward GMTM on a synthetic trimodal batch, plus a unimodal ablation.

The model is the same class the training scripts instantiate. The
hyperparameters default to a CPU-friendly slice of the experimental
``HParams`` (2 layers, 32-d, 4 heads). Pass ``--full`` to use the
4-layer / 64-d configuration from ``train_GMTM_bert.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))

from examples.repo import ensure_import_path

ensure_import_path()

import torch
import torch.nn.functional as F

from examples.shapes import BERT, GLOVE
from examples.synthetic_data import make_batch
from models import GatedMultiTransfomerModel


class DemoHParams:
    """Subset of the GMTM knobs the constructor actually reads."""

    def __init__(self, full: bool) -> None:
        self.num_heads = 4
        self.layers = 4 if full else 2
        self.attn_dropout = 0.1
        self.attn_dropout_modalities = [0.0, 0.0, 0.1]
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.2
        self.embed_dim = 64 if full else 32
        self.attn_mask = True
        self.output_dim = 1
        self.all_steps = False


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run_forward(model: GatedMultiTransfomerModel, batch, tag: str) -> torch.Tensor:
    pred = model(batch.as_list())
    loss = F.l1_loss(pred, batch.labels)
    print(
        f"  {tag:<28} pred={tuple(pred.shape)}  "
        f"L1={loss.item():.4f}  pred_range="
        f"[{pred.min().item():+.3f}, {pred.max().item():+.3f}]"
    )
    if pred.ndim != 2 or pred.shape[-1] != 1:
        raise RuntimeError(f"GMTM should return [B, 1], got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError("GMTM produced non-finite values")
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=("bert", "glove"), default="bert")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the 4-layer / 64-d HParams from train_GMTM_bert.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BERT if args.embedding == "bert" else GLOVE
    device = _device(args.device)
    hyp = DemoHParams(full=args.full)
    print(
        f"device={device}  embedding={args.embedding}  "
        f"layers={hyp.layers}  embed_dim={hyp.embed_dim}  heads={hyp.num_heads}"
    )

    torch.manual_seed(0)
    model = GatedMultiTransfomerModel(3, list(spec.as_tuple), hyp_params=hyp).to(device)
    model.eval()
    print(f"parameters: {_count_params(model):,}")

    batch = make_batch(spec, batch_size=args.batch_size, seq_len=args.seq_len, seed=7, device=device)
    print(f"inputs: {[tuple(x.shape) for x in batch.as_list()]}")

    with torch.no_grad():
        run_forward(model, batch, "trimodal")
        for keep in ("text", "audio", "visual"):
            dropped = [name for name in ("visual", "audio", "text") if name != keep]
            uni = batch.zero_modalities(dropped)
            run_forward(model, uni, f"unimodal {keep} (zeros)")

    weights = torch.softmax(model.modal_weights, dim=0).detach().cpu().tolist()
    print(
        "softmax modal_weights "
        f"(visual, audio, text) = {[round(w, 4) for w in weights]}"
    )
    print("GMTM forward pass is finite and shaped [B, 1].")


if __name__ == "__main__":
    main()
