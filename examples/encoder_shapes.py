"""Print input / output shapes for the unimodal encoders in models.py.

Useful when you change a hidden size and need the fusion `in_dim` or the
MLP head to follow. All modules run on the example device with a tiny
BERT-shaped batch.
"""

from __future__ import annotations

import torch

from common import count_parameters, device, feature_dims
from synthetic_data import make_corpus


def _import_encoders():
    from models import GRUWithLinear, LSTM, Transformer, TransformerSeq

    return LSTM, GRUWithLinear, Transformer, TransformerSeq


def _run_lstm(LSTM, name, x, lengths, in_dim, hid):
    packed = LSTM(in_dim, hid, dropout=True, has_padding=True).to(x.device)
    flat = LSTM(in_dim, hid, dropout=False, has_padding=False).to(x.device)
    y_packed = packed([x, lengths])
    y_flat = flat(x)
    print(f"{name} LSTM packed  {tuple(x.shape)} + lengths -> {tuple(y_packed.shape)}  params={count_parameters(packed)}")
    print(f"{name} LSTM raw     {tuple(x.shape)} -> {tuple(y_flat.shape)}")


def _run_gru(GRUWithLinear, name, x, lengths, in_dim, hid, out_dim):
    enc = GRUWithLinear(in_dim, hid, out_dim, dropout=True, has_padding=True).to(x.device)
    y = enc([x, lengths])
    print(f"{name} GRUWithLinear packed {tuple(x.shape)} -> {tuple(y.shape)}  params={count_parameters(enc)}")


def _run_transformers(Transformer, TransformerSeq, name, x, in_dim, hid):
    last = Transformer(in_dim, hid).to(x.device)
    seq = TransformerSeq(in_dim, hid).to(x.device)
    y_last = last(x)
    y_seq = seq(x)
    print(f"{name} Transformer (last step) {tuple(x.shape)} -> {tuple(y_last.shape)}  params={count_parameters(last)}")
    print(f"{name} TransformerSeq          {tuple(x.shape)} -> {tuple(y_seq.shape)}  params={count_parameters(seq)}")


def main() -> None:
    torch.manual_seed(0)
    LSTM, GRUWithLinear, Transformer, TransformerSeq = _import_encoders()
    corpus = make_corpus(n=3, text="bert", min_len=10, max_len=16, seed=4)
    features, lengths, _ = corpus.packed()
    names = ("visual", "audio", "text")
    dims = feature_dims("bert")
    hiddens = (64, 256, 1024)
    gru_outs = (19, 39, 159)  # TensorFusion BERT widths

    print(f"device: {device()}")
    print()
    for name, x, leng, in_dim, hid, gout in zip(names, features, lengths, dims, hiddens, gru_outs):
        print(f"--- {name}  F={in_dim} ---")
        _run_lstm(LSTM, name, x, leng, in_dim, hid)
        _run_gru(GRUWithLinear, name, x, leng, in_dim, hid, gout)
        # Smaller transformer width so the CPU example stays light.
        t_dim = 32 if in_dim < 100 else 64
        _run_transformers(Transformer, TransformerSeq, name, x, in_dim, t_dim)
        print()


if __name__ == "__main__":
    main()
