# Hyperparameters

Values below are the ones **written in the committed scripts**, not the
`train()` function defaults (those defaults are RMSprop + cross-entropy and
are overridden everywhere).

## Shared across MOSEI runs

| Knob | Value | Where |
| --- | --- | --- |
| Batch size | 32 | every `train_*.py` |
| Max sequence length | 50 | `max_seq_len` / `max_pad_num` |
| Optimizer | AdamW | `optimtype=torch.optim.AdamW` |
| Learning rate | `1e-4` | `lr=1e-4` |
| Weight decay | `0.01` | `weight_decay=0.01` |
| Loss | L1 (MAE) | `objective=torch.nn.L1Loss()` |
| Grad clip | 8 | `clip_val=8` in `train()` |
| Early stop patience | 7 | `early_stop=True` |
| Data workers | 0 | `num_workers=0` |
| `z_norm` | off | loader default |

## GMTM `HParams` (BERT and GloVe scripts are identical)

```python
class HParams:
    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]  # vis, aud, text
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

Unused fields (`modality_dropout`, `use_text_transformer`, `all_steps`,
`alpha`) are kept in the class because the scripts define them; the current
`forward` does not consume them.

Class-level `GatedMultiTransfomerModel.DefaultHyperParams` is a smaller
fallback (`embed_dim=9`, `heads=3`, `layers=3`) and is only used if you
construct GMTM without passing `hyp_params`.

## Baseline encoder widths (BERT)

| Fusion | Vision enc | Audio enc | Text enc | Fusion / head in |
| --- | --- | --- | --- | --- |
| ConcatEarly | Identity | Identity | Identity | LSTM(877 → 1024) + MLP(1024 → 1) |
| ConcatLate | LSTM 35→64 | LSTM 74→256 | LSTM 768→1024 | MLP(1344 → 1) |
| LowRankTensorFusion | GRU 35→64→32 | GRU 74→256→64 | GRU 768→1024→256 | LRTF rank 32, out 256 → MLP |
| TensorFusion | GRU →19 | GRU →39 | GRU →159 | TFN → MLP(128000 → 1) |
| TransformerEarly | Identity | Identity | Identity | EFT `n_features=877`, MLP(64→1) |
| TransformerLate | TSeq 35→64 | TSeq 74→128 | TSeq 768→1024 | LFT `in_dim=1216`, MLP(32→1) |

## Baseline encoder widths (GloVe)

| Fusion | Vision enc | Audio enc | Text enc | Fusion / head in |
| --- | --- | --- | --- | --- |
| ConcatEarly | Identity | Identity | Identity | LSTM(409 → 512) + MLP(512 → 1) |
| ConcatLate | LSTM 35→64 | LSTM 74→256 | LSTM 300→512 | MLP(832 → 1) |
| LowRankTensorFusion | GRU →32 | GRU →64 | GRU →128 | LRTF rank 32, out 128 → MLP |
| TensorFusion | GRU →19 | GRU →39 | GRU →79 | TFN → MLP(64000 → 1) |
| TransformerEarly | Identity | Identity | Identity | EFT `n_features=409`, Identity head |
| TransformerLate | TSeq 35→64 | TSeq 74→128 | TSeq 300→512 | LFT `in_dim=1792`, MLP(32→1) |

LSTM / GRU encoders use `dropout=True` (p=0.1) and `has_padding=True` except
on the early-transformer path.

## Epoch counts in the committed files

These look like leftover debug values — treat them as “what the file says
today,” not as the recipe that produced the CSVs:

| Script | `total_epochs` argument | `train()` currently |
| --- | ---: | --- |
| `train_main_bert.py` | 1 | enabled |
| `train_main_glove.py` | 10 | commented out |
| `train_GMTM_bert.py` | 50 | commented out |
| `train_GMTM_glove.py` | 20 | commented out |

If you re-fit, start from the GMTM numbers (20–50) with early stopping, not
from the `epochs=1` BERT sweep.

## Example / test hyperparameters

[`examples/common.py`](../examples/common.py) exposes a **tiny** `HParams`
(`embed_dim=16`, `layers=1`, `num_heads=2`) so CPU forward and the two-epoch
toy trainer finish in seconds. Comments in that file point back to the
64 / 4 / 4 training configuration.
