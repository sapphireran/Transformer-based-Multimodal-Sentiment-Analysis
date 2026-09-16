# Hyperparameters

Nothing in this repo reads a config file at train time. The numbers below
are copied out of the Python scripts so they can be cited from the docs
and from `examples/configs/*.yaml`.

## Shared across fusion sweeps

| Knob | Value | Where |
| --- | --- | --- |
| batch size | 32 | every `train_*.py` |
| `max_seq_len` / pad | 50 | loaders, GMTM |
| optimizer | AdamW | `train()` calls |
| learning rate | 1e-4 | `train()` calls |
| weight decay | 0.01 | `train()` calls |
| loss | L1 | `train()` / `test()` |
| early stop patience | 7 extra epochs | `train()` |
| grad clip | 8.0 | `train()` default |
| dataloader workers | 0 | scripts |
| `z_norm` | False | scripts |
| `robust_test` | False | scripts |

## Encoder widths — BERT (text = 768)

From `train_main_bert.py` / `mosi_test/train_mosi_bert.py`.

| Fusion | Vision enc | Audio enc | Text enc | Fused width | Head |
| --- | --- | --- | --- | ---: | --- |
| ConcatEarly | Identity | Identity | Identity | 877 (seq) | LSTM 877→1024 + MLP 1024→1 |
| ConcatLate | LSTM 35→64 | LSTM 74→256 | LSTM 768→1024 | 1344 | MLP 1344→1 |
| LMF | GRU+lin →32 | GRU+lin →64 | GRU+lin →256 | 256 (rank 32) | MLP 256→1 |
| TFN | GRU+lin →19 | GRU+lin →39 | GRU+lin →159 | 128000 | MLP 128000→2048→1 |
| TransformerEarly | Identity | Identity | Identity | 32 (last step) | MLP 64→64→1 |
| TransformerLate | TSeq 35→64 | TSeq 74→128 | TSeq 768→1024 | 32 (last step) | MLP 32→32→1 |

`TransformerSeq` / `LateFusionTransformer` / `EarlyFusionTransformer` all
use `nn.TransformerEncoderLayer(..., nhead=4)` and `num_layers=4`.
`EarlyFusionTransformer.embed_dim` is hard-coded to 32.

LSTM / GRU dropout is `dropout=True` (the modules default `dropoutp=0.1`).

## Encoder widths — GloVe (text = 300)

From `train_main_glove.py`.

| Fusion | Vision enc | Audio enc | Text enc | Fused width | Head |
| --- | --- | --- | --- | ---: | --- |
| ConcatEarly | Identity | Identity | Identity | 409 (seq) | LSTM 409→512 + MLP 512→1 |
| ConcatLate | LSTM 35→64 | LSTM 74→256 | LSTM 300→512 | 832 | MLP 832→1 |
| LMF | GRU+lin →32 | GRU+lin →64 | GRU+lin →128 | 128 (rank 32) | MLP 128→1 |
| TFN | GRU+lin →19 | GRU+lin →39 | GRU+lin →79 | 64000 | MLP 64000→2048→1 |
| TransformerEarly | Identity | Identity | Identity | 32 | Identity (head inside? script sets `head = Identity`; fusion returns 32-d) |
| TransformerLate | TSeq 35→64 | TSeq 74→128 | TSeq 300→512 | script `in_dim=1792` | MLP 32→32→1 |

The GloVe TransformerEarly path differs from BERT: BERT attaches
`MLP(64, 64, 1)`, GloVe attaches `Identity`. If you retrain GloVe early
fusion, decide which head you want and keep it consistent with the
checkpoint you load.

## GMTM `HParams` (both embeddings)

`train_GMTM_bert.py`, `train_GMTM_glove.py`, `mosi_test/mult_*_mosi.py`
share one class:

```python
class HParams:
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
    modality_dropout = 0.2          # stored, not read by GMTM.forward
    use_text_transformer = True     # stored, not read by GMTM.forward
```

Constructor:

```python
GatedMultiTransfomerModel(n_modalities=3, n_features=[35, 74, 768|300], hyp_params=HParams)
```

`attn_dropout_modalities[j]` is the attention dropout of the encoder that
uses modality `j` as key/value when `mem=False`. Index 2 (text) is the
only non-zero extra dropout.

`modality_dropout` and `use_text_transformer` are **not** referenced
inside `GatedMultiTransfomerModel`. They are leftovers from an earlier
draft; changing them in YAML will not change the module unless you wire
them up.

## DefaultHyperParams vs scripts

If you construct GMTM without passing `hyp_params`, you get
`embed_dim=9`, `num_heads=3`, `layers=3`, `embed_dropout=0.25`,
`out_dropout=0.0`. That combination is legal (`9 % 3 == 0`) and is what
`examples/run_gmtm_toy.py` uses so a CPU step stays small. Do not load a
64-d checkpoint into that constructor.

## Epoch counts as commented in the scripts

| Script | Commented / active epochs |
| --- | --- |
| `train_main_bert.py` | **active 1** (debug) |
| `train_main_glove.py` | commented 10 |
| `train_GMTM_bert.py` | commented 50 |
| `train_GMTM_glove.py` | commented 20 |

Published CSVs come from earlier runs, not from re-executing the file
with `total_epochs=1`.

## Randomness

There is no `torch.manual_seed` in the training scripts. A from-scratch
rerun will not bitwise-match the CSVs. Directional ranking (GMTM < late
transformer < LMF < concat on BERT-MOSEI MAE) is the reproducible claim,
not the fourth decimal.
