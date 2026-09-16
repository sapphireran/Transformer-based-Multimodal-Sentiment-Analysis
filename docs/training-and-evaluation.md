# Training and evaluation

The only training entry point is `train()` in `model/train_and_test.py`.
Every `train_*.py` script is a recipe: pick encoders / fusion / head, point
at a pickle, call `train` then `test`.

## Loop

```
for epoch in 1..total_epochs:
    model.train()
    for batch in train_dataloader:
        pred = MultiFramework(modalities)
        loss = L1(pred, label)          # deal_with_objective
        loss.backward()
        clip_grad_norm_(params, 8)
        optimizer.step()
    model.eval()
    valid_loss = mean L1 on valid
    if valid_loss improved:
        torch.save(model, save)         # full module, not state_dict
        patience = 0
    else:
        patience += 1
        if early_stop and patience > 7:
            break
```

Defaults from the BERT / GMTM scripts:

| Knob | Value |
| --- | --- |
| optimizer | `torch.optim.AdamW` |
| `lr` | `1e-4` |
| `weight_decay` | `0.01` |
| objective | `nn.L1Loss` (MAE) |
| `clip_val` | 8 |
| `early_stop` | True, patience 7 |
| `input_to_float` | True |
| `track_complexity` | True (`memory_profiler`) |

`train_main_bert.py` currently passes `total_epochs=1`. The GloVe / GMTM
scripts have the `train(...)` call **commented out** and only `torch.load`
a checkpoint. The numbers in `model/results/*.csv` come from earlier,
longer fits. To retrain, uncomment those blocks and raise `total_epochs`
(the commented GMTM BERT call used 50; GloVe used 20).

Checkpoints are the **entire** `MultiFramework` (`torch.save(model, path)`).
Loading needs the same class definitions in `sys.path` and, historically,
the same PyTorch pickle protocol. Prefer CPU `map_location` if you only
want to inspect weights.

## `deal_with_objective`

| Criterion | Target handling |
| --- | --- |
| `CrossEntropyLoss` | squeeze extra dim, `.long()` |
| `MSELoss`, `L1Loss`, `BCEWithLogitsLoss` | `.float()` |
| anything else | `objective(pred, truth, args)` |

Sentiment runs stay on `L1Loss`. Classification would need a different
head width and `CrossEntropyLoss`; nothing in the current sweeps does that.

## Test-time metrics

`single_test` collects every batch’s raw prediction `oute` (list of
`[score]`) and the labels, then computes:

### Regression

```
MSE  = mean((y − ŷ)²)
MAE  = mean(|y − ŷ|)
Corr = Pearson r(y, ŷ)          # scipy.stats.pearsonr
```

### Acc-7 and Acc-5 (uniform bins)

`split_uniform_7` / `split_uniform_5` slice **[-3, 3]** into equal-width
bins and run `numpy.digitize` (left-closed, right-open), clipping to
`{1..K}`. This is **not** the “hasuman” MOSI binning used in some papers
(those map `±3, ±2, ±1, 0` to seven integer classes). Numbers in this
repo’s CSVs are the uniform-bin version. `examples/04_metrics_walkthrough.py`
prints the edges.

### Acc-2 and F1

`eval_affect(truth, pred, exclude_zero=True)`:

1. Drop indices where the gold label is exactly 0 (optional).
2. Binarize both gold and prediction with `> 0`.
3. `accuracy_score` and binary `f1_score`.

The sign of the **regression** output is the class. There is no separate
classification head.

`single_test` also builds a matplotlib confusion matrix on the Acc-7 bins
and calls `plt.show()`. Headless example scripts use the helpers in
`examples/metrics.py` instead, which skip plotting.

## Packed vs. padded at train/test

`train(..., is_packed=)` and `test(..., is_packed=)` must match how the
batch was collated:

- packed (`_process_1`): `model([[mod.float() for mod in j[0]], j[1]])`
- padded (`_process_2`): `model([mod.float() for mod in j[:-1]])`

Mismatching this is the usual `TypeError` / shape error when you reuse a
dataloader with the wrong fusion.

## Complexity printout

With `track_complexity=True` (default):

```
Training Time: ...
Training Peak Mem: ...
Training Params: ...
```

`getallparams` sums `numel()` over the wrapped modules. `all_in_one_test`
prints inference time the same way. `test()` currently calls the inner
process **twice** (once inside `all_in_one_test`, once to capture the
dict) — the CSV writers only keep the second return value.

## What “good” looks like on MOSEI

On the checked-in BERT test split, GMTM all-three sits at MAE **0.564**,
Acc-2 **0.843**, Corr **0.726**. Fusion baselines cluster around
0.58–0.62 MAE. A run that lands near 0.8 MAE on MOSEI with text present is
usually a loader / packed-flag / checkpoint mismatch, not a weak fusion.

Audio-only and visual-only GMTM ablations are *supposed* to be weak
(~0.82 MAE, Corr 0.11–0.23). If they suddenly match text-only, the
ablation zeros are not being applied.
