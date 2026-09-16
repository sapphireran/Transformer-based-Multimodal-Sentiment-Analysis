# Training recipes

All full-scale trainers live under [`model/`](../model) and assume CUDA.
For a CPU walkthrough of the same GMTM wiring, use
[`examples/03_gmtm_tiny_train.py`](../examples/03_gmtm_tiny_train.py).

## Shared loop

[`train()`](../model/train_and_test.py) in `train_and_test.py`:

1. Wraps `(encoders, fusion, head)` in `MultiFramework`.
2. Optimizes with the caller-supplied optimizer (trainers use `AdamW`, `lr=1e-4`,
   `weight_decay=0.01`).
3. Clips gradients at `clip_val=8`.
4. Saves the **entire** `MultiFramework` object with `torch.save(model, path)`
   whenever validation L1 improves.
5. Optionally early-stops after 8 worse epochs.

`test()` loads that pickle (`torch.load(...).cuda()`) and runs `single_test`.

Because checkpoints are full objects, they are pickle-compatible with the
class definitions in `models.py` / `train_and_test.py`. Refactoring those
class names will break old `.pt` files.

`track_complexity=True` (default) wraps training in `memory_profiler.memory_usage`.
Install `memory-profiler` or the import at the top of `train_and_test.py` fails
even if you only want metrics.

## Script map

| Script | Embedding | Data | What it does |
| --- | --- | --- | --- |
| `train_main_bert.py` | BERT | MOSEI | Train six fusion baselines, write `main_results.csv` |
| `train_main_glove.py` | GloVe | MOSEI | Same, checkpoints named `glove_<Fusion>.pt` |
| `train_GMTM_bert.py` | BERT | MOSEI | GMTM ablation over modality subsets |
| `train_GMTM_glove.py` | GloVe | MOSEI | Same for GloVe |
| `mosi_test/train_mosi_bert.py` | BERT | MOSI merged | Load MOSEI fusion checkpoints, score MOSI |
| `mosi_test/train_mosi_glove.py` | GloVe | MOSI merged | Same (imports `training_structures.Supervised_Learning` — see below) |
| `mosi_test/mult_bert_mosi.py` | BERT | MOSI merged | Load GMTM ablation checkpoints, score MOSI |
| `mosi_test/mult_glove_mosi.py` | GloVe | MOSI merged | Same for GloVe GMTM |

Hyperparameters that are actually used in the BERT GMTM script:

```
epochs:        50 (commented train call) / test-only if you leave train() commented
batch_size:    32
max_seq_len:   50
max_pad:       True
num_workers:   0
optimizer:     AdamW
lr:            1e-4
weight_decay:  0.01
objective:     L1Loss
early_stop:    True
embed_dim:     64
layers:        4
num_heads:     4
```

Several trainers ship with `train(...)` **commented out** and only run
`torch.load` + `test`. Uncomment the train block when you need a new checkpoint.

## Checkpoint paths

| Experiment | Path pattern |
| --- | --- |
| MOSEI BERT fusion | `model/checkpoints/{ConcatEarly,ConcatLate,LowRankTensorFusion,TensorFusion,TransformerEarly,TransformerLate}.pt` |
| MOSEI GloVe fusion | `model/checkpoints/glove_{Fusion}.pt` |
| MOSEI BERT GMTM ablation | `model/checkpoints/ablation/model_{text+audio+visual}.pt` (plus subset names) |
| MOSEI GloVe GMTM ablation | `model/checkpoints/ablation/model_glove_{...}.pt` |

MOSI scripts prefix those paths with `../checkpoints/`.

### Naming traps

1. MOSI BERT fusion looks for `Concat.pt`, but the MOSEI trainer saves `ConcatLate.pt`.
   Copy or symlink if you want that row.
2. `train_mosi_glove.py` imports `from training_structures.Supervised_Learning import train, test`.
   That module is **not** in this repo (it is a MultiBench-style path). Prefer
   `from train_and_test import train, test` like the BERT MOSI script, or keep a
   local MultiBench install on `PYTHONPATH`.
3. `test(..., dataset='mosi', no_robust=True)` appears in some MOSI scripts, but
   the current `test()` signature only accepts
   `(model, test_dataloaders_all, is_packed, criterion, input_to_float)`.
   Extra keywords will `TypeError`. The BERT MOSI fusion script has this mismatch
   today; `examples/` and `model/metrics.py` do not.

## Packed vs. padded

| Method | `max_pad` | `is_packed` |
| --- | --- | --- |
| ConcatEarly / ConcatLate / TFN / LMF / TransformerLate | False | True |
| TransformerEarly | True | False |
| GMTM | True | False |

Mixing these (loading a packed checkpoint against a padded loader) will crash
inside `MultiFramework.forward`.

## Ablation protocol

`get_ablation_dataloader(..., modalities=['text', 'visual'])` keeps the three-encoder
GMTM graph and zeroes the unused channel. Compare rows only within the same
embedding (BERT vs GloVe) and the same dataset (MOSEI vs merged MOSI).

## Complexity helper

`all_in_one_train` / `all_in_one_test` print wall time, peak RSS, and parameter
count. They call the train/test closure **and** `test()` calls `_testprocess`
again afterward, so published inference-time numbers from this helper are
easy to double-count. Prefer timing a single `single_test` if you need a
clean number.

## Suggested local command

```bash
cd model
python train_GMTM_bert.py          # after uncommenting train()
cd mosi_test
python mult_bert_mosi.py           # MOSI transfer of that checkpoint
```

Next MOSI-native training (not transfer) is written up in [`mosi.md`](mosi.md).
