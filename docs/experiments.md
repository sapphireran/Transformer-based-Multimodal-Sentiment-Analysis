# Experiment scripts

All runnable training / eval entry points sit under `model/`. They
share one pattern:

1. Build dataloaders from a pickle (`get_dataloader` or an ablation
   variant).
2. Construct `encoders`, `fusion`, `head`.
3. `train(...)` then `torch.load` the best checkpoint.
4. `test(...)` → row in a CSV.

Working directory is expected to be `model/` so the relative pickle
and checkpoint paths resolve.

## Shared hyperparameters

| Item | Logged runs |
| --- | --- |
| Batch size | 32 |
| Max sequence length | 50 |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Objective | `L1Loss` |
| Early stop | patience 7 on val L1 |
| `num_workers` | 0 |
| Device | first CUDA device (scripts call `.cuda()`) |

`train_main_bert.py` currently passes `total_epochs=1` (useful as a
smoke run; the logged CSVs came from longer personal runs whose
checkpoints are not in git). Re-enable a larger epoch count and
`early_stop=True` before treating a new CSV as comparable.

## MOSEI BERT baselines — `train_main_bert.py`

Pickle: `data/MOSEI/mosei_raw_bert.pkl` (`[35, 74, 768]`).

Two loaders are built up front:

- `*_OT`: `max_pad=False` → packed sequences for LSTM/GRU fusions.
- `*_TE`: `max_pad=True` → fixed 50-step tensors for TransformerEarly.

| `fusion_method` | Encoders | Fusion | Head | Packed? |
| --- | --- | --- | --- | --- |
| ConcatEarly | 3× Identity | `ConcatEarly` | LSTM(877→1024) + MLP | yes |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | `ConcatLate` | MLP(1344→1) | yes |
| LowRankTensorFusion | GRUWithLinear → 32/64/256 | LRTF rank 32, out 256 | MLP(256→1) | yes |
| TensorFusion | GRUWithLinear → 19/39/159 | `TensorFusion` | MLP(128000→2048→1) | yes |
| TransformerEarly | 3× Identity | `EarlyFusionTransformer(877)` | MLP(64→1) | no |
| TransformerLate | TransformerSeq 64/128/1024 | `LateFusionTransformer(1216)` | MLP(32→1) | yes |

Checkpoint path: `checkpoints/{fusion_method}.pt`.
CSV: `main_results.csv` (a copy also lives in `results/`).

## MOSEI GloVe baselines — `train_main_glove.py`

Same six names, widths swapped to `[35, 74, 300]`:

- ConcatEarly LSTM input becomes `409`.
- ConcatLate text LSTM is `300→512`, head `832`.
- LRTF factors `[32, 64, 128]`, head `128`.
- TensorFusion text GRU projects to `64`, head dim changes accordingly.
- Early transformer `n_features=409`.
- Late transformer `in_dim` is the concatenated unimodal widths.

CSV: `glove_results.csv`.

## MOSEI GMTM + modality ablation — `train_GMTM_bert.py` / `train_GMTM_glove.py`

GMTM always sees three tensors. Unused modalities are zeroed by
`get_ablation_dataloader(..., modalities=..., embedding='bert'|'glove')`.

```
encoders = [Identity, Identity, Identity]
fusion   = GatedMultiTransfomerModel(3, input_dims, hyp_params=HParams)
head     = Identity
is_packed = False
max_pad   = True
```

`HParams`: 4 heads, 4 layers, `embed_dim=64`, text attn dropout 0.1,
embed dropout 0.2, output dropout 0.1.

The combination list is the power set of `{text, audio, visual}` minus
the empty set (7 rows). BERT script currently leaves only the full
triple uncommented; GloVe script still loops all seven.

Checkpoints (when training is uncommented):

```
checkpoints/ablation/model_{text+audio+visual}.pt
checkpoints/ablation/model_glove_{text+audio+visual}.pt
```

The `train(...)` calls are commented in the checked-in copies; the
scripts load those filenames and run `test`. Uncomment training to
reproduce.

## MOSI eval scripts — `model/mosi_test/`

These load **MOSEI-trained** checkpoints from `../checkpoints/` and
score them on MOSI pickles. That is transfer, not a MOSI-from-scratch
train loop.

| Script | Embedding | Loader | Output CSV |
| --- | --- | --- | --- |
| `train_mosi_bert.py` | BERT | `get_mosi_dataloader` | `mosi_bert_results.csv` |
| `train_mosi_glove.py` | GloVe | same | `mosi_glove_results.csv` |
| `mult_bert_mosi.py` | BERT GMTM | `get_ablation_mosi_dataloader` | printed / appended |
| `mult_glove_mosi.py` | GloVe GMTM | same | same |

`get_mosi_dataloader` **merges train+valid+test** (see
[`datasets.md`](datasets.md)). The MOSI CSVs are therefore not the
standard MOSI test split.

Naming drift: MOSI late-concat is saved/loaded as `Concat` in some
scripts and `ConcatLate` in others. Match the filename under
`checkpoints/` before running.

## Checkpoints

`model/checkpoints/readme.md` and `checkpoints/ablation/readme.md` are
location stubs. The `.pt` files themselves are not in git (they are
full `torch.save(model)` pickles and are machine-specific).

`torch.save(model, path)` stores the class definition path. Loading on
a newer PyTorch may require `weights_only=False` (PyTorch 2.6+).

## Complexity logging

When `track_complexity=True` (the default in `train`),
`all_in_one_train` wraps the loop in `memory_profiler.memory_usage`
and prints:

- wall-clock training time
- peak RSS
- parameter count (`getallparams`)

`all_in_one_test` prints inference time and param count. The function
currently calls the test body **twice** (once inside the timer, once
to return the dict). That doubles the printed inference time and the
confusion-matrix popup.

## What not to run from a laptop

TensorFusion's 128k-D head is the memory hog. GMTM with `embed_dim=64`
and 9×4-layer encoders is the compute hog. The CPU examples under
`examples/` use `embed_dim=16`, `layers=1` for that reason.

## Suggested personal rerun order

1. `examples/run_all.py` — architecture + metrics sanity, no data.
2. `train_GMTM_bert.py` with training uncommented,  one seed, full
   three modalities — confirms the pickle and GPU path.
3. The 7-way ablation on BERT, then GloVe.
4. Baselines in `train_main_bert.py` with a real epoch budget.
5. MOSI transfer scripts only after the MOSEI checkpoints exist.
