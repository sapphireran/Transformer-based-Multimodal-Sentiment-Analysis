# Experiments

This page is a wiring diagram for the original scripts. It does not re-run
them. Recorded CSVs are treated as the source of truth; several scripts have
the `train(...)` call commented out and only load a checkpoint.

Always launch from `model/` so the pickle paths and `sys.path` hacks resolve.

```bash
cd model
```

## Shared training hyperparameters

From `train()` defaults and the call sites:

| Knob | Value used in the sweeps |
| --- | --- |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Objective | `L1Loss` (MAE) |
| Epochs (BERT baselines) | `1` in the committed `train_main_bert.py` — raise this for a real fit |
| Epochs (commented GMTM BERT) | `50` with `early_stop=True` (patience 7) |
| Epochs (commented GMTM GloVe) | `20` |
| Epochs (commented GloVe baselines) | `10` |
| Grad clip | `8` |
| Batch size | `32` |
| `num_workers` | `0` |
| Early stopping | on; stop after 7 epochs without a new best val loss |
| Checkpoint | `torch.save(model, path)` of the whole `MultiFramework` |

`track_complexity=True` by default, which wraps the loop in
`memory_profiler.memory_usage` and prints peak RAM plus parameter count.

## MOSEI baseline sweep

| Script | Text | Output CSV | Checkpoint pattern |
| --- | --- | --- | --- |
| `train_main_bert.py` | BERT 768 | `main_results.csv` | `checkpoints/{Fusion}.pt` |
| `train_main_glove.py` | GloVe 300 | `glove_results.csv` | `checkpoints/glove_{Fusion}.pt` |

Fusion names, in order:

1. `ConcatEarly`
2. `ConcatLate` (GloVe / BERT MOSEI) — MOSI eval scripts use the name `Concat`
   for the same late-concat checkpoint
3. `LowRankTensorFusion`
4. `TensorFusion`
5. `TransformerEarly`
6. `TransformerLate`

`TransformerEarly` is the only baseline that sets `max_pad=True` and
`is_packed=False`. Everything else uses variable-length packed sequences.

GloVe hidden sizes (from `train_main_glove.py`):

| Fusion | Differences vs BERT |
| --- | --- |
| ConcatEarly | `LSTM(409, 512)` + `MLP(512,512,1)` |
| ConcatLate | text LSTM `300→512`, head `MLP(832,832,1)` |
| LMF | text GRU `300→128`, fusion out 128 |
| TFN | text GRU out 79, head `MLP(64000,2048,1)` |
| TransformerEarly | `n_features=409`, head is `Identity` (the transformer already ends in 32-d; **no extra MLP**) |
| TransformerLate | text `TransformerSeq(300, 512)`, `in_dim=1792` as written in the script |

`in_dim=1792` does not equal `64+128+512=704`. If you revive that run, check
the `LateFusionTransformer` projection against the actual concatenated encoder
width — the examples use the arithmetic width so the CPU walkthrough is
consistent.

`train_main_glove.py` currently **does not train**; it `torch.load`s
`glove_{Fusion}.pt`. Uncomment the `train(...)` block to fit.

## GMTM + modality ablation (MOSEI)

| Script | Text | Combinations in the committed file | Checkpoints |
| --- | --- | --- | --- |
| `train_GMTM_bert.py` | BERT | only `['text','audio','visual']` (others commented) | `checkpoints/ablation/model_{a+b+c}.pt` |
| `train_GMTM_glove.py` | GloVe | all 7 subsets | `checkpoints/ablation/model_glove_{a+b+c}.pt` |

Both build:

```python
encoders = [Identity(), Identity(), Identity()]
fusion   = GatedMultiTransfomerModel(3, [35, 74, text_dim], hyp_params=HParams)
head     = Identity()
```

with `HParams.embed_dim = 64`, 4 heads, 4 layers. Ablation zeros unused
streams rather than shrinking the module (see [data-pipeline.md](data-pipeline.md)).

Training is commented out in both GMTM scripts; they load the `.pt` and call
`test()`. `train_GMTM_glove.py` still writes `ablation_glove_results.csv`.
`train_GMTM_bert.py` has the CSV write commented out — the table in
`model/results/ablation_results.csv` is the preserved copy.

`train_GMTM_glove.py` passes `dataset=` / `no_robust=` into `test()`, but the
current `train_and_test.test()` signature does not accept those keywords. If
you run that file as-is against this tree it will `TypeError`. The BERT GMTM
script already uses the matching signature.

## MOSI transfer (`model/mosi_test/`)

These scripts do not train. They load a MOSEI checkpoint and score a MOSI
pickle through `get_mosi_dataloader` / `get_ablation_mosi_dataloader`.

| Script | What it scores | Writes |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT baselines | `mosi_bert_results.csv` |
| `train_mosi_glove.py` | GloVe baselines | `mosi_glove_results.csv` |
| `mult_bert_mosi.py` | GMTM BERT (full trio only) | write is commented out |
| `mult_glove_mosi.py` | GMTM GloVe, 7 subsets | `ablation_mosi_glove_results.csv` |

Caveats:

- MOSI loaders **merge train+valid+test** (not the official test set).
- `train_mosi_glove.py` imports `training_structures.Supervised_Learning`,
  which is **not** in this repository. It looks like a leftover MultiBench
  path. Prefer `train_and_test` (as the BERT MOSI script does) if you revive
  it.
- Fusion name `Concat` in the MOSI BERT script maps to checkpoint
  `../checkpoints/Concat.pt`, while MOSEI training saves `ConcatLate.pt`.
  Rename or symlink if you want that row to load.

## Suggested reproduction order

1. Build or copy `mosei_raw_bert.pkl` into `model/data/MOSEI/`.
2. Uncomment `train(...)` in `train_main_bert.py` and raise `total_epochs`
   from `1` to something like `20–50`.
3. Run the six baselines; confirm `main_results.csv` is rewritten.
4. Uncomment training in `train_GMTM_bert.py`, enable the seven modality
   lists, run, write `ablation_results.csv`.
5. Repeat with the GloVe pickle if you care about the embedding ablation.
6. Only then point `mosi_test/` at the new checkpoints, and decide whether
   you want the merged-MOSI protocol or a true test-only loader.

## CPU stand-ins that do run in this checkout

No pickle, no checkpoint, no CUDA:

```bash
python3 examples/fusion_shapes.py
python3 examples/gmtm_forward.py
python3 examples/multiframework_demo.py
python3 examples/read_result_tables.py
```

Those instantiate the same classes with the same feature widths and print
output shapes / recorded tables.
