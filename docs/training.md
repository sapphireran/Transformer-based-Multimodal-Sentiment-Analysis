# Training

All real experiments go through `train()` / `test()` in
`model/train_and_test.py`. This page is the operator's view: which
script to run, what it writes, and which knobs actually change the
run.

## Scripts

Run them from `model/` so the relative pickle and checkpoint paths
resolve.

| Script | Data | What it does |
| --- | --- | --- |
| `train_main_bert.py` | `data/MOSEI/mosei_raw_bert.pkl` | Sweeps the six fusion baselines. **Calls `train()`** (1 epoch as checked in). Writes `main_results.csv` and `checkpoints/{Fusion}.pt`. |
| `train_main_glove.py` | `data/MOSEI/mosei_raw_glove.pkl` | Same sweep, GloVe widths. `train()` is commented out; it only loads `checkpoints/glove_{Fusion}.pt`. |
| `train_GMTM_bert.py` | same BERT pickle | GMTM. Ablation list is currently only `['text', 'audio', 'visual']`. `train()` is commented out; loads `checkpoints/ablation/model_text+audio+visual.pt`. |
| `train_GMTM_glove.py` | GloVe pickle | Full 7-way modality ablation. `train()` commented out; loads `model_glove_{mod+...}.pt`. |
| `mosi_test/train_mosi_bert.py` | `data/MOSI/mosi_raw_bert.pkl` | Scores MOSEI BERT checkpoints on concatenated MOSI. |
| `mosi_test/train_mosi_glove.py` | MOSI GloVe pickle | Same for GloVe. Still imports `training_structures.Supervised_Learning` — that is a leftover MultiBench path and will fail unless you add that package or point the import at `train_and_test`. |
| `mosi_test/mult_bert_mosi.py` | MOSI BERT pickle | GMTM transfer, BERT. |
| `mosi_test/mult_glove_mosi.py` | MOSI GloVe pickle | GMTM transfer, GloVe; writes `ablation_mosi_glove_results.csv`. |

Recorded CSVs that match a finished run already live in
`model/results/` and `model/mosi_test/`. Re-running a test-only script
overwrites the copy next to the script, not necessarily the one under
`results/`.

## Shared hyperparameters

`train()` defaults and the values the BERT main sweep actually passes:

| Knob | `train()` default | Sweep / GMTM value |
| --- | --- | --- |
| optimizer | `RMSprop` | `AdamW` |
| `lr` | `1e-3` | `1e-4` |
| `weight_decay` | `0` | `0.01` |
| `objective` | `CrossEntropyLoss` | `L1Loss` |
| `clip_val` | 8 | 8 |
| `early_stop` | `False` | `True` (patience 7 on val L1) |
| `total_epochs` | caller | `1` in `train_main_bert.py`; commented GMTM calls used 20–50 |
| `batch_size` | — | 32 |
| `num_workers` | — | 0 |
| `is_packed` | `False` | `False` for TransformerEarly / GMTM, `True` otherwise |

`input_to_float=True` casts every modality to `float32` before the
forward. Labels stay whatever the collate function built (a float
`[B, 1]` for MOSI/MOSEI).

`track_complexity=True` wraps the train closure in
`memory_profiler.memory_usage` and prints wall time, peak RSS, and
parameter count. Install `memory-profiler` or pass
`track_complexity=False` if you do not want that dependency.

## How a training step looks

```text
for batch in train_dataloader:
    if is_packed:
        out = model([[x.float().to(device) for x in batch[0]], batch[1]])
    else:
        out = model([x.float().to(device) for x in batch[:-1]])
    loss = L1(out, batch[-1])
    loss.backward()
    clip_grad_norm_(model.parameters(), 8)
    optimizer.step()
```

Packed batches from `_process_1` are
`(list_of_3_tensors, list_of_3_lengths, indices, labels)`. The
`is_packed` branch only sends the tensors and the lengths;
`MultiFramework` then calls each encoder as
`encoder([tensor, length])`.

Max-pad batches from `_process_2` are
`(vision, audio, text, labels)`. `batch[:-1]` is exactly the three
modalities GMTM / TransformerEarly expect.

After each epoch the loop scores validation L1 (no metric suite). If
the value improved, it `torch.save`s the **entire `MultiFramework`
module** (not a `state_dict`) to `save=`. `torch.load` later must be
able to resolve `MultiFramework` and every nested class, so keep
`models.py` / `train_and_test.py` on `sys.path` when you reload.

## Early stopping

Patience is hard-coded at 7 epochs with no improvement in mean
validation L1. There is no warmup and no learning-rate schedule. If
you uncomment the GMTM `train(..., 50)` call, a run can stop well
before 50 epochs.

## Checkpoint layout

```text
model/checkpoints/
    ConcatEarly.pt
    ConcatLate.pt
    LowRankTensorFusion.pt
    TensorFusion.pt
    TransformerEarly.pt
    TransformerLate.pt
    glove_ConcatEarly.pt
    ...
    ablation/
        model_text+audio+visual.pt
        model_glove_text.pt
        model_glove_text+audio+visual.pt
        ...
```

The filenames are part of the test scripts' contract. If you train a
new ablation combo, keep the `'+'.join(modalities)` order
(`text`, `audio`, `visual` as written in the list, not alphabetical).

Weights are **not** in git (see `.gitignore`). The `readme.md` files
under `checkpoints/` only mark the directory.

## Uncommenting a fresh GMTM train

In `train_GMTM_bert.py`:

```python
train(
    encoders, fusion, head, traindata, validdata, 50,
    optimtype=torch.optim.AdamW, early_stop=True, is_packed=False,
    lr=1e-4,
    save=f"checkpoints/ablation/model_{'+'.join(modalities)}.pt",
    weight_decay=0.01, objective=torch.nn.L1Loss(),
)
```

Use `get_ablation_dataloader(..., max_pad=True, embedding='bert')` so
the zero-filled dropped modalities have the BERT widths. Switch
`embedding='glove'` and `input_dims = [35, 74, 300]` for the GloVe
twin. Restore the commented 7-way `modality_combinations` list to
reproduce `ablation_results.csv`.

## Device notes

Training scripts call `.cuda()` at construction time. On a CPU-only
machine that throws before the first batch. The `examples/` demos
force CPU and never call `.cuda()`. For a CPU smoke test of the real
scripts, replace `.cuda()` with `.to(device)` and set
`device = torch.device("cpu")`.

`torch.load(path).cuda()` on a checkpoint saved from GPU works on GPU;
on CPU use `torch.load(path, map_location="cpu")`.

## Logging

There is no TensorBoard / W&B hook. The loop prints:

- per-epoch mean train L1
- per-epoch mean valid L1
- `"Saving Best Model"` when validation improves
- after `test()`, MAE / Acc-7 / Acc-5 / Acc-2 / Corr / F1

`single_test` also opens a matplotlib confusion matrix via
`plt.show()`. On a headless server that can block. The example metric
helpers skip plotting; for a real eval on a server, wrap `test()` in
`MPLBACKEND=Agg` or comment out the `plt.show()` block.

## Suggested first real run

1. Confirm `data/MOSEI/mosei_raw_bert.pkl` loads and
   `drop_entry` leaves a non-empty train split.
2. Run **one** fusion, not the whole sweep. In `train_main_bert.py`
   set `fusion_methods = ['ConcatLate']` and raise `total_epochs`
   from 1 to at least 10.
3. Only then uncomment GMTM training. GMTM's 9 cross-modal encoders
   are the slow path.
