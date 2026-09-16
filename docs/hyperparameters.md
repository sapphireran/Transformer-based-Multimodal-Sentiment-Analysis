# Hyperparameters

Values below are what the scripts actually pass, not a suggested
search grid. When a line is commented out in a file, I still record
the last uncommented setting I can see next to it.

## Shared training defaults

From `train()` in [`model/train_and_test.py`](../model/train_and_test.py)
unless a script overrides them.

| Knob | Default | Recorded override |
| --- | --- | --- |
| Optimizer | `RMSprop` | `AdamW` in every sweep script |
| `lr` | `1e-3` | `1e-4` |
| `weight_decay` | `0` | `0.01` |
| Objective | `CrossEntropyLoss` | `L1Loss` |
| `clip_val` | `8` | unchanged |
| `early_stop` | `False` | `True` (patience 8) |
| `input_to_float` | `True` | unchanged |
| `is_packed` | `False` | `True` except TransformerEarly / GMTM |
| Batch size | — | `32` |
| Max seq len | — | `50` |
| `num_workers` | `2` | `0` in the scripts |

`train_main_bert.py` currently passes `total_epochs=1`. That looks like
a debug leftover; the GloVe script's commented call used `10`, GMTM
BERT used `50`, GMTM GloVe used `20`. Re-enable a real epoch count
before treating a new CSV as comparable.

## GMTM `HParams`

Duplicated in `train_GMTM_bert.py`, `train_GMTM_glove.py`, and
`model/mosi_test/mult_bert_mosi.py`. Class defaults in
`GatedMultiTransfomerModel.DefaultHyperParams` are different (smaller);
the scripts always pass this override.

| Field | Script value | Class default |
| --- | ---: | ---: |
| `num_heads` | 4 | 3 |
| `layers` | 4 | 3 |
| `attn_dropout` | 0.1 | 0.1 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` | 1000 zeros |
| `relu_dropout` | 0.1 | 0.1 |
| `res_dropout` | 0.1 | 0.1 |
| `out_dropout` | 0.1 | 0.0 |
| `embed_dropout` | 0.2 | 0.25 |
| `embed_dim` | 64 | 9 |
| `attn_mask` | True | True |
| `output_dim` | 1 | 1 |
| `all_steps` | False | False |
| `modality_dropout` | 0.2 | (absent) |
| `use_text_transformer` | True | (absent) |

`modality_dropout` and `use_text_transformer` are stored on the HParams
class and **never read** by `GatedMultiTransfomerModel`. Cross-modal
dropout is `attn_dropout_modalities[j]` on the key/value side.

`num_heads` must divide `embed_dim` (64 / 4 = 16). The class default
`embed_dim=9`, `num_heads=3` also divides.

## Fusion encoder widths (BERT)

| Method | Encoders | Fusion | Head |
| --- | --- | --- | --- |
| ConcatEarly | 3 × Identity | concat dim 2 → 877 | LSTM 877→1024, MLP 1024→1 |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | concat → 1344 | MLP 1344→1 |
| LowRankTensorFusion | GRU+lin → 32 / 64 / 256 | rank 32, out 256 | MLP 256→1 |
| TensorFusion | GRU+lin → 19 / 39 / 159 | outer product 128000 | MLP 128000→2048→1 |
| TransformerEarly | 3 × Identity | EarlyFusionTransformer(877) → 32 | MLP 64→1 **or** Identity |
| TransformerLate | TransformerSeq 35→64, 74→128, 768→1024 | LateFusionTransformer(1216) → 32 | MLP 32→1 |
| GMTM | 3 × Identity | GMTM(3, [35, 74, 768]) | Identity |

TransformerEarly's `MLP(64, 64, 1)` does not match the fusion output
width of 32. The BERT script still constructs that head. If you hit a
matmul error on a fresh run, change the head to `MLP(32, 32, 1)` or
`Identity` + a linear inside the fusion module.

## Fusion encoder widths (GloVe)

| Method | Encoders | Fusion | Head |
| --- | --- | --- | --- |
| ConcatEarly | 3 × Identity | concat → 409 | LSTM 409→512, MLP 512→1 |
| ConcatLate | LSTM 35→64, 74→256, 300→512 | concat → 832 | MLP 832→1 |
| LowRankTensorFusion | GRU+lin → 32 / 64 / 128 | rank 32, out 128 | MLP 128→1 |
| TensorFusion | GRU+lin → 19 / 39 / 79 | outer product 64000 | MLP 64000→2048→1 |
| TransformerEarly | 3 × Identity | EarlyFusionTransformer(409) | Identity |
| TransformerLate | TransformerSeq 35→64, 74→128, 300→512 | `in_dim=1792` in the script | MLP 32→1 |
| GMTM | 3 × Identity | GMTM(3, [35, 74, 300]) | Identity |

See [architecture.md](architecture.md) for the 1792 vs 704 mismatch.

## Synthetic example knobs

Used only by `examples/` so a laptop / CPU agent can finish in seconds.

| Knob | Value |
| --- | --- |
| `B` | 8 |
| `T` | 12 |
| `F_v, F_a, F_t` | 8, 10, 16 |
| GMTM `embed_dim` | 16 |
| GMTM `layers` | 2 |
| GMTM `num_heads` | 4 |
| Toy train steps | 25 |
| Toy lr | `3e-3` |
