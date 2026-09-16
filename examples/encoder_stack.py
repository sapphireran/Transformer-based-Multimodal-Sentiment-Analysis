"""Run the per-modality encoders the baseline scripts stack in front of fusion."""

from __future__ import annotations

import torch

from examples.common import BERT_LAYOUT, cpu_device, finite, format_shape
from examples.synthetic_data import make_batch


def run_encoders(seq_len: int = 10, batch_size: int = 3) -> list[dict]:
    from models import GRUWithLinear, LSTM, MLP, Transformer, TransformerSeq

    device = cpu_device()
    batch = make_batch(batch_size=batch_size, seq_len=seq_len)
    vision = batch.vision.to(device)
    audio = batch.audio.to(device)
    text = batch.text.to(device)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    cases = []

    lstm = LSTM(BERT_LAYOUT.visual, 16, dropout=False, has_padding=False)
    out = lstm(vision)
    cases.append(("LSTM(vision, hid=16)", format_shape(out), finite(out)))

    packed = LSTM(BERT_LAYOUT.audio, 16, dropout=False, has_padding=True)
    out = packed([audio, lengths])
    cases.append(("LSTM packed(audio)", format_shape(out), finite(out)))

    gru = GRUWithLinear(BERT_LAYOUT.text, 32, 8, dropout=False, has_padding=False, batch_first=True)
    out = gru(text)
    cases.append(("GRUWithLinear(text → 8)", format_shape(out), finite(out)))

    tseq = TransformerSeq(BERT_LAYOUT.visual, 16)
    out = tseq(vision)
    cases.append(("TransformerSeq(vision → 16)", format_shape(out), finite(out)))

    tlast = Transformer(BERT_LAYOUT.audio, 16)
    out = tlast(audio)
    cases.append(("Transformer last-step(audio)", format_shape(out), finite(out)))

    mlp = MLP(BERT_LAYOUT.text, 32, 1)
    out = mlp(text.mean(1))
    cases.append(("MLP(mean text → 1)", format_shape(out), finite(out)))

    for name, shape, ok in cases:
        if not ok:
            raise RuntimeError(f"{name} produced non-finite values")
    return [{"name": n, "shape": s} for n, s, _ in cases]


def main() -> int:
    print("Baseline encoder output ranks (synthetic BERT batch, T=10)")
    for row in run_encoders():
        print(f"  {row['name']:<34} {row['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
