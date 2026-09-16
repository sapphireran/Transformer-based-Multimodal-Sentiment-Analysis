# Training

The shared loop is `train()` / `test()` / `single_test()` in
[`model/train_and_test.py`](../model/train_and_test.py). Each
experiment script only chooses encoders, fusion, head, pickle path,
and whether the batch is packed.

## MultiFramework

```
inputs  →  encoder_0, encoder_1, encoder_2
        →  fusion(list_of_encoded)
        →  head(fused)
        →  [B, 1]  sentiment score
```

`has_padding=True` passes `[features, lengths]` into each encoder
(packed LSTM / GRU). `has_padding=False` passes the raw `[B, T, F]`
tensor. GMTM and TransformerEarly always use the unpacked path.

## Default hyperparameters

From the BERT / GloVe bake-off scripts:

| Knob | Value |
| --- | --- |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Objective | `torch.nn.L1Loss` (MAE) |
| Batch size | 32 |
| Early stop | patience 7 on validation L1 |
| Grad clip | 8 |
| Device | `cuda:0` if available else CPU |
| Checkpoint | overwrite `best.pt` / a named path on every val improvement |

`train_main_bert.py` currently sets `total_epochs=1` (a smoke-debug
leftover). The commented GMTM calls use 20–50 epochs. Re-run with a
real epoch budget if you want to match the CSVs.

`input_to_float=True` casts every modality to `float32` before the
forward pass. Labels stay whatever the collate function produced;
`deal_with_objective` casts them to float for L1 / MSE / BCE.

## Script map

Run these from `model/` so relative pickle and checkpoint paths
resolve.

| Script | Data | What it does | Writes |
| --- | --- | --- | --- |
| `train_main_bert.py` | MOSEI BERT | 6 fusion methods | `main_results.csv`, `checkpoints/{method}.pt` |
| `train_main_glove.py` | MOSEI GloVe | 6 fusion methods | `glove_results.csv`, `checkpoints/glove_{method}.pt` |
| `train_GMTM_bert.py` | MOSEI BERT | GMTM, optional ablations | `ablation_results.csv`, `checkpoints/ablation/model_{mods}.pt` |
| `train_GMTM_glove.py` | MOSEI GloVe | GMTM ablations | `ablation_glove_results.csv` |
| `mosi_test/train_mosi_bert.py` | MOSI BERT test | load MOSEI checkpoints | `mosi_bert_results.csv` |
| `mosi_test/train_mosi_glove.py` | MOSI GloVe test | load MOSEI checkpoints | `mosi_glove_results.csv` |
| `mosi_test/mult_bert_mosi.py` | MOSI BERT | GMTM transfer + ablation | `ablation_mosi_results.csv` |
| `mosi_test/mult_glove_mosi.py` | MOSI GloVe | GMTM transfer + ablation | `ablation_mosi_glove_results.csv` |

Copies of the CSVs also live under `model/results/` so the numbers
survive a working-directory overwrite.

MOSI loaders merge train+valid+test into one evaluation set
(`get_mosi_dataloader`). That is intentional: those scripts measure
how a MOSEI-trained checkpoint behaves on every MOSI clip, not a
held-out MOSI split you trained on.

## BERT vs GloVe encoder widths

The fusion graphs are the same idea; only the text tower changes.

**BERT** (`text=768`, early total 877):

- ConcatLate LSTMs: 35→64, 74→256, 768→1024; head `MLP(1344, 1344, 1)`
- LRTF GRUs: 32 / 64 / 256, rank 32, head `MLP(256, 256, 1)`
- TFN GRUs: 19 / 39 / 159, head `MLP(128000, 2048, 1)`
- TransformerLate: `TransformerSeq` 64 / 128 / 1024, late `in_dim=1216`

**GloVe** (`text=300`, early total 409):

- ConcatLate LSTMs: 35→64, 74→256, 300→512; head `MLP(832, 832, 1)`
- LRTF GRUs: 32 / 64 / 128, head `MLP(128, 128, 1)`
- TFN GRUs: 19 / 39 / 79, head `MLP(64000, 2048, 1)`
- TransformerLate: `TransformerSeq` 64 / 128 / 512

GMTM always uses Identity encoders and `input_dims = [35, 74, text_dim]`.

## Ablations

`get_ablation_dataloader(..., modalities=['text', 'audio'])` keeps
those streams and replaces the others with zeros of the correct
`(50, F)` shape. GMTM still sees three inputs; a zero stream just
contributes nothing useful after the projection. The combinations
logged in the CSVs are:

```
['text']
['audio']
['visual']
['text', 'audio']
['text', 'visual']
['audio', 'visual']
['text', 'audio', 'visual']
```

`examples/zero_modality_ablation.py` repeats that pattern on synthetic
data so you can see the API without the pickles.

## What the examples do instead

`examples/train_toy_gmtm.py` does **not** call `train()`. It runs a
short AdamW + L1 loop on `make_synthetic_split()` so you can step
through GMTM on CPU in a few seconds. Checkpointing, early stop, and
`memory_profiler` are omitted on purpose.

If you want to exercise `MultiFramework` itself, start from
`examples/forward_fusion.py`, which builds each fusion graph with the
BERT widths and prints output shapes.
