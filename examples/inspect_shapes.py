"""Print tensor shapes through the BERT-sized stack (short time axis).

Uses the real modules from ``model/models.py`` with a batch of 2 and
``T=6`` so the printout stays readable. Feature widths are the MOSI/MOSEI
ones (35 / 74 / 768).
"""

from __future__ import annotations

import torch

from common import AUDIO_DIM, TEXT_BERT_DIM, VISUAL_DIM, device
from synthetic_multimodal import random_batch

import models as M


def _line(label: str, tensor: torch.Tensor) -> None:
    print(f"  {label:28s} {tuple(tensor.shape)}")


def main() -> int:
    dev = device()
    torch.manual_seed(0)
    v, a, t = random_batch(batch_size=2, seq_len=6, text_dim=TEXT_BERT_DIM, device=dev)
    print(f"device={dev}")
    print("raw batch")
    _line("vision", v)
    _line("audio", a)
    _line("text (BERT width)", t)

    print("\nConcatEarly")
    fused = M.ConcatEarly()([v, a, t])
    _line("concat features", fused)

    print("\nConcatLate (LSTM summaries)")
    hv = M.LSTM(VISUAL_DIM, 16).to(dev)(v)
    ha = M.LSTM(AUDIO_DIM, 24).to(dev)(a)
    ht = M.LSTM(TEXT_BERT_DIM, 32).to(dev)(t)
    _line("LSTM vision", hv)
    _line("LSTM audio", ha)
    _line("LSTM text", ht)
    late = M.ConcatLate()([hv, ha, ht])
    _line("late concat", late)

    print("\nTensorFusion on last-step GRU projections")

    def last_gru(indim, hid, out, x):
        y = M.GRUWithLinear(indim, hid, out, batch_first=True).to(dev)(x)
        return y[:, -1, :] if y.dim() == 3 else y

    tv, ta, tt = (
        last_gru(VISUAL_DIM, 16, 8, v),
        last_gru(AUDIO_DIM, 24, 12, a),
        last_gru(TEXT_BERT_DIM, 32, 16, t),
    )
    _line("proj vision", tv)
    _line("proj audio", ta)
    _line("proj text", tt)
    tfn = M.TensorFusion()([tv, ta, tt])
    _line("TFN outer product", tfn)
    print(f"  expected width               {(8+1)*(12+1)*(16+1)}")

    print("\nLowRankTensorFusion rank=4 → 16")
    lmf = M.LowRankTensorFusion([8, 12, 16], 16, rank=4).to(dev)
    _line("LMF", lmf([tv, ta, tt]))

    print("\nTransformerEarly (n_features=877)")
    early = M.EarlyFusionTransformer(n_features=VISUAL_DIM + AUDIO_DIM + TEXT_BERT_DIM).to(dev)
    _line("last step", early([v, a, t]))

    print("\nTransformerLate (16+24+32)")
    s_v = M.TransformerSeq(VISUAL_DIM, 16).to(dev)(v)
    s_a = M.TransformerSeq(AUDIO_DIM, 24).to(dev)(a)
    s_t = M.TransformerSeq(TEXT_BERT_DIM, 32).to(dev)(t)
    _line("TransformerSeq vision", s_v)
    _line("TransformerSeq audio", s_a)
    _line("TransformerSeq text", s_t)
    late_tr = M.LateFusionTransformer(in_dim=16 + 24 + 32, embed_dim=16).to(dev)
    _line("late transformer last", late_tr([s_v, s_a, s_t]))

    print("\nGMTM (embed_dim=16, 1 layer)")

    class H:
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

    gmtm = M.GatedMultiTransfomerModel(3, [VISUAL_DIM, AUDIO_DIM, TEXT_BERT_DIM], hyp_params=H).to(dev)
    _line("ŷ", gmtm([v, a, t]))
    print("\nshape inspection passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
