# Training

All real training goes through `train()` in `model/train_and_test.py`. The
function always wraps your pieces in `MultiFramework` and moves that module
to `cuda:0` if CUDA exists, otherwise CPU.

## Call signature (what the scripts actually pass)

```python
train(
    encoders, fusion, head,
    train_dataloader, valid_dataloader,
    total_epochs,
    optimtype=torch.optim.AdamW,
    early_stop=True,
    is_packed=...,          # False only for TransformerEarly / GMTM
    lr=1e-4,
    save="checkpoints/....pt",
    weight_decay=0.01,
    objective=torch.nn.L1Loss(),
)
```

Unchecked defaults you inherit if you call `train()` yourself:

| Argument | Default | Meaning |
| --- | --- | --- |
| `clip_val` | `8` | global grad-norm clip |
| `input_to_float` | `True` | cast modality tensors to float |
| `track_complexity` | `True` | wrap the loop in `memory_profiler` + param count |
| `early_stop` patience | `7` epochs without a new best val loss | only if `early_stop=True` |
| `optimtype` (library default) | `RMSprop` | scripts override to `AdamW` |

Best checkpoint = lowest **validation L1**. The file is a full `torch.save`
of the `MultiFramework` instance.

## Loss

`deal_with_objective` special-cases:

- `CrossEntropyLoss` — squeezes a trailing class dim and casts labels to
  `long`
- `MSELoss`, `L1Loss`, `BCEWithLogitsLoss` — casts labels to `float`
- anything else — `objective(pred, truth, args)`

The sentiment experiments use **`L1Loss`** (MAE). Predictions stay a
`(B, 1)` real number. Acc7 / Acc2 are **not** training targets.

## Per-epoch loop

1. `model.train()`, iterate the train loader with `tqdm`.
2. Forward:
   - packed: `model([[x.float().to(device) for x in batch[0]], batch[1]])`
   - unpacked: `model([x.float().to(device) for x in batch[:-1]])`
3. Backward, clip, optimizer step.
4. `model.eval()`, no-grad validation L1.
5. Save if `valloss` improved.

`track_complexity=True` runs that whole nested function under
`memory_usage`, then prints wall time, peak RAM, and parameter count. That
can make a one-epoch debug run look slower than it is.

## Script-by-script

### `train_main_bert.py` (MOSEI BERT fusion sweep)

- Data: `data/MOSEI/mosei_raw_bert.pkl`, `batch_size=32`
- Builds **two** loader triples: packed (`traindata_OT`) and max-padded
  (`traindata_TE`)
- Methods: ConcatEarly, ConcatLate, LowRankTensorFusion, TensorFusion,
  TransformerEarly, TransformerLate
- `total_epochs=1` in the checked-in file — bump this before a real run
- Writes `main_results.csv` in the current working directory

### `train_main_glove.py`

Same methods, GloVe pickle, different hidden sizes (see
[fusion_methods.md](fusion_methods.md)). The `train(...)` call is commented
out; the script only `torch.load`s `checkpoints/glove_{method}.pt`. Uncomment
and set epochs (the commented block uses `10`) to retrain.

### `train_GMTM_bert.py` / `train_GMTM_glove.py`

- `get_ablation_dataloader(..., max_pad=True, modalities=...)`
- Identity encoders + GMTM + Identity head
- BERT script’s train call (50 epochs) is commented; GloVe’s (20 epochs) is
  too
- GloVe walks all 7 modality subsets; BERT is currently hard-coded to the
  full trio

### MOSI scripts (`model/mosi_test/`)

Evaluation only. They load MOSEI checkpoints and score MOSI. There is no
MOSI training loop in the checked-in files.

## Device notes

Scripts call `.cuda()` on modules at construction time. On a CPU-only
machine that raises. The examples under `examples/` use a helper that picks
CPU or CUDA and never assume a GPU.

`LowRankTensorFusion` builds the homogeneous-1 column on
`cuda:0` if `torch.cuda.is_available()` else CPU. Mixing a CPU model with
that path is fine; mixing CUDA modules with CPU tensors is not.

## Recommended personal workflow

1. Confirm the pickle loads and print one batch
   (`examples/packed_vs_padded.py` is the synthetic version of this).
2. Overfit a **tiny** subset with ConcatLate, 20–50 steps, to prove labels
   match tensors.
3. Run the fusion sweep with a real epoch count (the historical tables used
   early stopping, not a single epoch).
4. Train GMTM with the full modality triple, then the six ablations if you
   care about stream contribution.
5. Optionally score MOSI with the same checkpoint, remembering the merged
   split (see [datasets.md](datasets.md)).

## Hyperparameter cheat sheet

Copied from the scripts, not from a sweep:

| Knob | Fusion sweep | GMTM |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| Learning rate | `1e-4` | `1e-4` |
| Weight decay | `0.01` | `0.01` |
| Batch size | 32 | 32 |
| Max length (padded) | 50 | 50 |
| Loss | L1 | L1 |
| Early stop | yes (patience 7) | yes |
| GMTM `embed_dim` | — | 64 |
| GMTM layers / heads | — | 4 / 4 |

Dropout lives inside the individual encoders (`dropout=True` on LSTM/GRU)
and inside GMTM (`embed_dropout=0.2`, `out_dropout=0.1`).

## Outputs

| Artifact | Produced by |
| --- | --- |
| `checkpoints/{Method}.pt` | BERT fusion `train()` |
| `checkpoints/glove_{Method}.pt` | GloVe fusion |
| `checkpoints/ablation/model_{a+b+c}.pt` | GMTM BERT |
| `checkpoints/ablation/model_glove_{...}.pt` | GMTM GloVe |
| `main_results.csv` / `glove_results.csv` | fusion sweeps |
| `ablation_results.csv` / `ablation_glove_results.csv` | GMTM ablations |

Copies of the CSVs also sit in `model/results/` and are transcribed in
[results.md](results.md).

## Complexity helpers

`getallparams`, `all_in_one_train`, and `all_in_one_test` print:

- training wall time and peak memory
- parameter count
- inference wall time

They do not write those numbers into the result CSVs. If you want them
alongside MAE, log them yourself.

## Failure modes seen in this codebase

| Symptom | Likely cause |
| --- | --- |
| `LSTM` `pack_padded_sequence` type error | `is_packed=False` on a packed loader, or the reverse |
| `Conv1d` channel mismatch | `LateFusionTransformer(in_dim=...)` ≠ sum of encoder dims |
| `MLP` `mat1/mat2` mismatch | head `indim` ≠ fusion output width (early transformer 32 vs 64) |
| `torch.load` `ModuleNotFoundError` | checkpoint pickled a class you renamed, or you loaded from the wrong cwd |
| Immediate `exit()` in `__getitem__` | an utterance with empty text slipped past `drop_entry` |
| MOSI numbers look “too easy/hard” vs a paper | merged train+valid+test loader |

`examples/gmtm_toy_train.py` is a CPU stand-in for step 2 above: it overfits
GMTM to synthetic labels so you can test the backward path without a pickle.
