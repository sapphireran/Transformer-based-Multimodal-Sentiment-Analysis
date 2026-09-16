# Code map

Personal index of every tracked source file and the one thing it is for.

## Repository root

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, headline table, synthetic quick start |
| `LICENSE` | MIT, copyright 2024 pang990801 |
| `requirements.txt` | PyTorch stack for examples and full training |
| `docs/` | The notes you are reading |
| `examples/` | CPU demos that import `model/models.py` |

## `model/models.py`

Single module file. Contains, in order of appearance:

- `Identity`, `Transformer`, `TransformerSeq`
- `LSTM`, `MLP`, `GRU`, `GRUWithLinear`
- `Linear` **class** (later shadowed)
- `TransformerFusion` (stack-of-vectors transformer; unused by sweeps)
- `ConcatLate`, `ConcatEarly`
- `TensorFusion`, `LowRankTensorFusion`
- `Linear` **factory** (Xavier uniform) — this is the name that remains
- `EarlyFusionTransformer`, `LateFusionTransformer`
- `TransformerEncoder`, `TransformerEncoderLayer` (GMTM internals)
- `fill_with_neg_inf`, `buffered_future_mask` (mask helpers; mask unused)
- `CustomLinear`, `CustomLayerNorm` (unused by sweeps)
- `make_positions`, `SinusoidalPositionalEmbedding`
- `AttentionPooling`
- `GatedMultiTransfomerModel`

## `model/train_and_test.py`

- `eval_affect` — Acc-2 / F1
- `MultiFramework` — encoder / fusion / head wrapper
- `deal_with_objective` — loss adapter
- `all_in_one_train` / `all_in_one_test` — time + memory + param count
- `train` — the only fit loop
- `split_uniform_7` / `split_uniform_5`
- `single_test` / `test`

## `model/data/`

| File | Purpose |
| --- | --- |
| `get_dataloader.py` | `Affectdataset`, four public loader functions, two collates |
| `readme.md` | Where `.csd` / GloVe files were stored locally |
| `analysis_dataset.ipynb` | Personal EDA notebook |
| `MOSEI/get_mosei.py` | SDK alignment scratch (name vs. contents: MOSI) |
| `MOSEI/get_dataset.ipynb` | Notebook form of the alignment |
| `MOSI/get_dataset.ipynb` | Same for MOSI |

## Training recipes (`model/`)

| Script | Dataset | Text | Trains? (as checked in) |
| --- | --- | --- | --- |
| `train_main_bert.py` | MOSEI | BERT | yes, 1 epoch, six fusions |
| `train_main_glove.py` | MOSEI | GloVe | no, load `checkpoints/glove_*.pt` |
| `train_GMTM_bert.py` | MOSEI | BERT | no, load ablation checkpoint |
| `train_GMTM_glove.py` | MOSEI | GloVe | no, load `model_glove_*` |

## MOSI transfer (`model/mosi_test/`)

| Script | Text | Loads from |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT | `../checkpoints/{Fusion}.pt` |
| `train_mosi_glove.py` | GloVe | `../checkpoints/glove_{Fusion}.pt` |
| `mult_bert_mosi.py` | BERT | `../checkpoints/ablation/model_{mods}.pt` |
| `mult_glove_mosi.py` | GloVe | `../checkpoints/ablation/model_glove_{mods}.pt` |

The `train_mosi_*` names are historical; they only test.

## Checkpoints and results

| Path | Purpose |
| --- | --- |
| `model/checkpoints/readme.md` | “main experiment `.pt` files go here” |
| `model/checkpoints/ablation/readme.md` | GMTM `.pt` files go here |
| `model/results/*.csv` | canonical personal tables |
| `model/results/plot.ipynb` | plots those tables |
| `model/*.csv` | copies of the same tables next to the scripts |

No `.pt` weights are in git (and `.gitignore` keeps it that way).

## Examples (`examples/`)

| Script | Imports from `model/` | Side effects |
| --- | --- | --- |
| `common.py` | path setup only | none |
| `metrics.py` | none (reimplements Acc-7/5/2) | none |
| `01_synthetic_batch.py` | no | prints shapes |
| `02_fusion_forward.py` | fusion modules | one forward each |
| `03_gmtm_forward.py` | `GatedMultiTransfomerModel` | one forward + param count |
| `04_metrics_walkthrough.py` | `metrics.py` only | prints bin edges and scores |
| `05_mini_training.py` | ConcatLate + LSTM + `train`-like loop | 8 tiny epochs on synthetic data |
| `06_result_tables.py` | none (stdlib csv) | ranks the git CSVs by MAE |

## Call graph (MOSEI BERT fusion sweep)

```
train_main_bert.py
  ├─ get_dataloader(mosei_raw_bert.pkl)          # packed
  ├─ get_dataloader(..., max_pad=True)           # TransformerEarly
  └─ run_experiment(name)
       ├─ models.{Concat*, *TensorFusion, *Transformer*}
       ├─ train_and_test.train
       │    └─ MultiFramework.forward
       └─ train_and_test.test → single_test → eval_affect
```

## Call graph (GMTM)

```
train_GMTM_bert.py
  └─ get_ablation_dataloader(..., modalities)
       └─ filter_modalities_list   # zeros unused streams
  encoders = Identity × 3
  fusion   = GatedMultiTransfomerModel(3, [35, 74, 768], HParams)
  head     = Identity
  train / torch.load + test   # is_packed=False
```
