"""Forward-pass every fusion used in the MOSEI BERT sweep.

No checkpoints and no pickles. Encoders are replaced by a last-step pool so
the fusion modules themselves are what get exercised. Run:

    python examples/fusion_forward.py
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List

import torch
import torch.nn as nn

from common import (
    AUDIO_DIM,
    BERT_DIM,
    VISUAL_DIM,
    format_stats,
    make_synthetic_batch,
    pool_last,
    tensor_stats,
)


def _load_models():
    from models import (
        ConcatEarly,
        ConcatLate,
        EarlyFusionTransformer,
        LateFusionTransformer,
        LowRankTensorFusion,
        TensorFusion,
        TransformerSeq,
    )

    return {
        "ConcatEarly": ConcatEarly,
        "ConcatLate": ConcatLate,
        "TensorFusion": TensorFusion,
        "LowRankTensorFusion": LowRankTensorFusion,
        "EarlyFusionTransformer": EarlyFusionTransformer,
        "LateFusionTransformer": LateFusionTransformer,
        "TransformerSeq": TransformerSeq,
    }


def run_concat_early(mods, batch) -> torch.Tensor:
    fused = mods["ConcatEarly"]()(batch.as_list)
    # Same head family as train_main_bert: LSTM on the concatenated stream
    # would sit here. We only check the fusion tensor.
    return fused


def run_concat_late(mods, batch) -> torch.Tensor:
    return mods["ConcatLate"]()(pool_last(batch.as_list))


def run_tensor_fusion(mods, batch) -> torch.Tensor:
    # BERT TensorFusion encoders project to 19 / 39 / 159. Use those widths
    # so the Kronecker size matches the published MLP (128000).
    vis, aud, txt = pool_last(batch.as_list)
    vis = nn.Linear(VISUAL_DIM, 19)(vis)
    aud = nn.Linear(AUDIO_DIM, 39)(aud)
    txt = nn.Linear(batch.text_dim, 159)(txt)
    return mods["TensorFusion"]()([vis, aud, txt])


def run_low_rank(mods, batch) -> torch.Tensor:
    vis, aud, txt = pool_last(batch.as_list)
    vis = nn.Linear(VISUAL_DIM, 32)(vis)
    aud = nn.Linear(AUDIO_DIM, 64)(aud)
    txt = nn.Linear(batch.text_dim, 256)(txt)
    return mods["LowRankTensorFusion"]([32, 64, 256], 256, 32)([vis, aud, txt])


def run_transformer_early(mods, batch) -> torch.Tensor:
    return mods["EarlyFusionTransformer"](n_features=batch.early_width)(batch.as_list)


def run_transformer_late(mods, batch) -> torch.Tensor:
    # Mirror train_main_bert.py: unimodal TransformerSeq then LateFusion.
    enc_v = mods["TransformerSeq"](VISUAL_DIM, 64)
    enc_a = mods["TransformerSeq"](AUDIO_DIM, 128)
    enc_t = mods["TransformerSeq"](batch.text_dim, 1024)
    seqs = [enc_v(batch.vision), enc_a(batch.audio), enc_t(batch.text)]
    return mods["LateFusionTransformer"](in_dim=64 + 128 + 1024)(seqs)


RUNNERS: Dict[str, Callable] = {
    "ConcatEarly": run_concat_early,
    "ConcatLate": run_concat_late,
    "TensorFusion": run_tensor_fusion,
    "LowRankTensorFusion": run_low_rank,
    "TransformerEarly": run_transformer_early,
    "TransformerLate": run_transformer_late,
}

# Expected trailing feature dim on a BERT batch (None = sequence tensor).
EXPECTED_LAST_DIM = {
    "ConcatEarly": VISUAL_DIM + AUDIO_DIM + BERT_DIM,  # [B, T, 877]
    "ConcatLate": VISUAL_DIM + AUDIO_DIM + BERT_DIM,  # last-step cat = 877
    "TensorFusion": 20 * 40 * 160,  # 128000
    "LowRankTensorFusion": 256,
    "TransformerEarly": 32,
    "TransformerLate": 32,
}


def run_all(batch_size: int = 4, seed: int = 0) -> List[Dict]:
    torch.manual_seed(seed)
    mods = _load_models()
    batch = make_synthetic_batch(batch_size=batch_size, embedding="bert", seed=seed)
    rows = []
    for name, runner in RUNNERS.items():
        out = runner(mods, batch)
        stats = tensor_stats(name, out)
        expected = EXPECTED_LAST_DIM[name]
        if out.size(-1) != expected:
            raise AssertionError(f"{name}: last dim {out.size(-1)} != {expected}")
        if not stats["finite"]:
            raise AssertionError(f"{name}: non-finite values")
        rows.append({"fusion": name, "output": out, **stats})
        print(format_stats(stats))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("Synthetic BERT batch → fusion modules (CPU)\n")
    rows = run_all(batch_size=args.batch_size, seed=args.seed)
    print(f"\n{len(rows)} fusions produced finite tensors.")


if __name__ == "__main__":
    main()
