"""Forward-pass GatedMultiTransfomerModel on synthetic BERT and GloVe batches.

Mirrors ``train_GMTM_bert.py`` / ``train_GMTM_glove.py``: three Identity
encoders, GMTM as the fusion, Identity head. Also shows the zeroed-modality
ablation the dataloader uses.

    python examples/gmtm_forward.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import torch

from common import format_stats, make_synthetic_batch, tensor_stats, zero_modalities


class HParams:
    """Same knobs as the GMTM train scripts."""

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
    modality_dropout = 0.2
    use_text_transformer = True


def build_gmtm(input_dims: Sequence[int]):
    from models import GatedMultiTransfomerModel

    return GatedMultiTransfomerModel(3, list(input_dims), hyp_params=HParams)


def score_batch(model, streams: List[torch.Tensor]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(streams)


def run_one(embedding: str, keep: Tuple[str, ...], batch_size: int, seed: int) -> Dict:
    batch = make_synthetic_batch(batch_size=batch_size, embedding=embedding, seed=seed)
    model = build_gmtm([35, 74, batch.text_dim])
    streams = zero_modalities(batch, keep)
    out = score_batch(model, streams)
    if out.shape != (batch_size, 1):
        raise AssertionError(f"GMTM {embedding} {keep}: shape {tuple(out.shape)} != ({batch_size}, 1)")
    stats = tensor_stats(f"GMTM-{embedding}-{'+'.join(keep)}", out)
    if not stats["finite"]:
        raise AssertionError(f"GMTM {embedding} {keep}: non-finite output")
    n_params = sum(p.numel() for p in model.parameters())
    return {"embedding": embedding, "keep": keep, "n_params": n_params, "output": out, **stats}


def run_suite(batch_size: int = 2, seed: int = 0) -> List[Dict]:
    # layers=4, 3x3 cross-modal stacks — keep the demo batch small.
    plans = [
        ("bert", ("text", "audio", "visual")),
        ("bert", ("text",)),
        ("glove", ("text", "audio", "visual")),
        ("glove", ("audio", "visual")),
    ]
    rows = []
    print("Gated Multi-Transformer on synthetic clips (CPU, eval mode)\n")
    for embedding, keep in plans:
        row = run_one(embedding, keep, batch_size, seed)
        rows.append(row)
        print(f"{format_stats(row)}  params={row['n_params']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run_suite(batch_size=args.batch_size, seed=args.seed)
    print(f"\n{len(rows)} GMTM configurations produced a [B, 1] score.")


if __name__ == "__main__":
    main()
