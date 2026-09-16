# Training

Personal runbook for the scripts under `model/`. All commands below assume the current working directory is `model/` and that the aligned pickles already exist (see [datasets](datasets.md)).

## Script matrix

| Script | Data | Text | What it does today |
| --- | --- | --- | --- |
| `train_main_bert.py` | MOSEI | BERT | Trains 6 fusion baselines for **1 epoch**, writes `checkpoints/<Fusion>.pt` + `main_results.csv` |
| `train_main_glove.py` | MOSEI | GloVe | Loads `checkpoints/glove_<Fusion>.pt` (train call commented), writes `glove_results.csv` |
| `train_GMTM_bert.py` | MOSEI | BERT | Ablation hook; train call commented; loads `checkpoints/ablation/model_text+audio+visual.pt` |
| `train_GMTM_glove.py` | MOSEI | GloVe | Same pattern for all 7 modality subsets |
| `mosi_test/train_mosi_bert.py` | MOSI (merged) | BERT | Evaluates MOSEI BERT checkpoints on MOSI |
| `mosi_test/train_mosi_glove.py` | MOSI (merged) | GloVe | Same for GloVe; still imports `training_structures.Supervised_Learning` |
| `mosi_test/mult_bert_mosi.py` | MOSI (merged) | BERT | GMTM transfer test |
| `mosi_test/mult_glove_mosi.py` | MOSI (merged) | GloVe | GMTM GloVe transfer test |

The GloVe MOSI tester is the oldest file in the set: it still talks to MultiBench's `training_structures.Supervised_Learning`. The BERT MOSI tester already uses the local `train_and_test.py`. Prefer the local module.

## Shared recipe

From `train()` in `train_and_test.py` and the BERT main script:

| Knob | Default in `train()` | Value used by BERT main |
| --- | --- | --- |
| Optimizer | `RMSprop` | `AdamW` |
| Learning rate | `1e-3` | `1e-4` |
| Weight decay | `0` | `0.01` |
| Objective | `CrossEntropyLoss` | `L1Loss` |
| Epochs | caller | **1** in `train_main_bert.py` (raise this) |
| Early stop | off | on, patience 7 (irrelevant at 1 epoch) |
| Grad clip | 8 | 8 |
| Batch size | — | 32 |
| `num_workers` | — | 0 |
| Device | first CUDA device else CPU | scripts call `.cuda()` unconditionally |

`all_in_one_train` wraps the loop with `memory_profiler.memory_usage` and prints wall time, peak RSS, and parameter count. Disable with `track_complexity=False` if you do not have `memory_profiler`.

## Per-fusion encoder / head (BERT / MOSEI)

Copied from `train_main_bert.py`. Input dims `[35, 74, 768]`.

| Fusion | Encoders | Fusion module | Head | Packed? |
| --- | --- | --- | --- | --- |
| ConcatEarly | 3× Identity | `ConcatEarly` | `LSTM(877, 1024) → MLP(1024, 1024, 1)` | yes |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | `ConcatLate` | `MLP(1344, 1344, 1)` | yes |
| LowRankTensorFusion | GRUWithLinear → 32 / 64 / 256 | LRTF `[32,64,256] → 256`, rank 32 | `MLP(256, 256, 1)` | yes |
| TensorFusion | GRUWithLinear → 19 / 39 / 159 | `TensorFusion` | `MLP(128000, 2048, 1)` | yes |
| TransformerEarly | 3× Identity | `EarlyFusionTransformer(877)` | `MLP(64, 64, 1)` | **no** (`max_pad`) |
| TransformerLate | `TransformerSeq` 35→64, 74→128, 768→1024 | `LateFusionTransformer(1216)` | `MLP(32, 32, 1)` | yes |

GloVe (`train_main_glove.py`) uses text width 300 and smaller heads (late concat 832, LRTF 128, tensor-fusion 64000, early transformer `n_features=409`). `TransformerLate` is constructed with `in_dim=1792` even though `64+128+512=704` — if you revive that run, set `in_dim` to the real concatenated width or the `Conv1d` will not match the encoder outputs.

## GMTM recipe

```python
encoders = [Identity(), Identity(), Identity()]
fusion   = GatedMultiTransfomerModel(3, [35, 74, 768 or 300], hyp_params=HParams)
head     = Identity()   # GMTM already ends in a 1-d head
```

Always three towers. Ablations zero a modality instead of removing a tower ([architecture](architecture.md)).

Suggested real run (the call is commented in the file):

```python
train(
    encoders, fusion, head, traindata, validdata, 50,   # or 20 for GloVe
    optimtype=torch.optim.AdamW,
    early_stop=True,
    is_packed=False,
    lr=1e-4,
    save=f"checkpoints/ablation/model_{'+'.join(modalities)}.pt",
    weight_decay=0.01,
    objective=torch.nn.L1Loss(),
)
```

Checkpoint names the ablation loader expects:

```text
checkpoints/ablation/model_text.pt
checkpoints/ablation/model_audio.pt
checkpoints/ablation/model_visual.pt
checkpoints/ablation/model_text+audio.pt
checkpoints/ablation/model_text+visual.pt
checkpoints/ablation/model_audio+visual.pt
checkpoints/ablation/model_text+audio+visual.pt
checkpoints/ablation/model_glove_text+audio+visual.pt   # GloVe prefix
```

## Packed vs padded in one place

```python
# packed (concat / tensor / late transformer)
traindata, validdata, testdata = get_dataloader(
    filepath, batch_size=32, data_type='mosei', num_workers=0
)
train(..., is_packed=True, ...)

# padded (early transformer + GMTM)
traindata, validdata, testdata = get_dataloader(
    filepath, batch_size=32, max_pad=True, data_type='mosei', num_workers=0
)
train(..., is_packed=False, ...)
```

Mixing the two (`is_packed=True` on a `_process_2` batch, or the reverse) will either index-error or silently train on garbage. `train_main_bert.py` already branches on `fusion == 'TransformerEarly'`.

## Saving and loading

`train()` does `torch.save(model, save)` on the **whole `MultiFramework`**, not a `state_dict`. Load with:

```python
model = torch.load(path, map_location="cpu")   # add weights_only=False on torch>=2.6
```

That is why the MOSI testers can load a MOSEI file and call `test()` without rebuilding encoders. It also means checkpoints are pickle-coupled to the current class layout in `models.py` / `train_and_test.py`. Do not rename `MultiFramework` if you still need old `.pt` files.

## Things to fix before a serious rerun

1. **Epochs.** `train_main_bert.py` uses `total_epochs=1`. That will not reproduce the published CSVs. Use something in the 20–50 range with `early_stop=True`.
2. **Uncomment `train(...)`** in the GloVe / GMTM scripts. They currently only evaluate.
3. **Strip unknown kwargs.** `test()` does not take `dataset=` or `no_robust=`. `train_main_glove.py` and `train_GMTM_glove.py` still pass them.
4. **`.cuda()`.** The scripts assume a GPU. For CPU, drop the `.cuda()` calls or wrap with `torch.device`.
5. **`train_mosi_glove.py`** still imports MultiBench. Point it at `train_and_test` like the BERT sibling.
6. **GloVe late-transformer `in_dim`.** 1792 vs 704, as above.

## Complexity printout

A successful `train(..., track_complexity=True)` ends with:

```text
Training Time: ...
Training Peak Mem: ...
Training Params: ...
```

`test()` prints inference time and parameter count, then the metric block from [evaluation](evaluation.md).
