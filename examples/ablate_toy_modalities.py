#!/usr/bin/env python3
"""Zero-out modalities on a frozen tiny GMTM, matching the ablation loaders.

``get_ablation_dataloader`` keeps three input slots and writes zeros into the
dropped ones. This script does the same on a synthetic batch so you can see
how a randomly initialized GMTM reacts before any MOSEI training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_model_to_path, ensure_output_dir  # noqa: E402
from gmtm_forward import TinyHParams  # noqa: E402
from toy_data import AUDIO_DIM, BERT_DIM, GLOVE_DIM, VISION_DIM, make_toy_batch  # noqa: E402

add_model_to_path()

from metrics import compute_sentiment_metrics, format_metrics  # noqa: E402

COMBINATIONS: List[Sequence[str]] = (
    ("text",),
    ("audio",),
    ("visual",),
    ("text", "audio"),
    ("text", "visual"),
    ("audio", "visual"),
    ("text", "audio", "visual"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bert", "glove"), default="glove")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--write-json", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from models import GatedMultiTransfomerModel

    torch.manual_seed(args.seed)
    text_dim = BERT_DIM if args.backend == "bert" else GLOVE_DIM
    full = make_toy_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        text_backend=args.backend,
        seed=args.seed,
    )
    model = GatedMultiTransfomerModel(
        n_modalities=3,
        n_features=[VISION_DIM, AUDIO_DIM, text_dim],
        hyp_params=TinyHParams,
    ).to(args.device)
    model.eval()

    rows = []
    print(f"random GMTM  backend={args.backend}  N={args.batch_size}")
    print()
    for keep in COMBINATIONS:
        batch = full.masked(keep)
        vision = torch.from_numpy(batch.vision).to(args.device)
        audio = torch.from_numpy(batch.audio).to(args.device)
        text = torch.from_numpy(batch.text).to(args.device)
        with torch.no_grad():
            pred = model([vision, audio, text])
        scores = compute_sentiment_metrics(batch.labels, pred)
        name = "+".join(keep)
        rows.append({"modalities": name, **{k: float(v) for k, v in scores.items()}})
        print(f"== {name} ==")
        print(format_metrics(scores))
        print()

    if args.write_json:
        out = ensure_output_dir() / "ablate_toy_modalities.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
