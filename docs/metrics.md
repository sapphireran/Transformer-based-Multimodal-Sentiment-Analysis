# Evaluation protocol

Sentiment here is a **regression** problem. The trainer minimizes
`L1Loss` (`MAE`) against the continuous label. After inference,
`single_test` in `model/train_and_test.py` also reports classification
numbers obtained by **binning** that continuous value.

This page restates the functions as they are implemented, including the
places they differ from the most common MOSI/MOSEI papers.

## Regression metrics

Let `y` be the true score and `ŷ` the model output, both shape `[N]`
after a squeeze.

| Name | Formula in code | Notes |
| --- | --- | --- |
| Test loss | mean of the criterion (L1 during the logged runs) | printed as `TestLoss` |
| MSE | `mean((y - ŷ)²)` | not written to the CSVs |
| MAE | `mean(\|y - ŷ\|)` | primary regression number |
| Corr | Pearson `r` via `scipy.stats.pearsonr` | linear association, not MAE |

`examples/metrics_walkthrough.py` recomputes MAE / MSE / Corr with
numpy so you can see the arithmetic without loading a checkpoint.

## Binary Acc-2 and F1 (`eval_affect`)

```
non_zeros = indices where y != 0          # exclude_zero=True (default)
binary_truth = (y[non_zeros] > 0)
binary_preds = (ŷ[non_zeros] > 0)
Acc2 = accuracy_score(binary_truth, binary_preds)
F1   = f1_score(..., average="binary")
```

Properties that matter when you compare to other papers:

1. **Neutral labels (`y == 0`) are dropped** before Acc-2 / F1.
2. The decision threshold is **0 on the raw regression output**, not a
   trained classifier.
3. `average="binary"` is the positive-class F1 (positive = `y > 0`).
   Many MOSI papers report *weighted* F1 of the two non-neutral classes
   instead. The numbers in `model/results/*.csv` are this function, not
   the weighted variant.
4. A second, unused mapping in `single_test` turns `ŷ` into
   `{+1, 0, -1}` for printing; Acc-2 / F1 do **not** use that mapping.
   They use the raw `ŷ` versus 0.

## Acc-7 and Acc-5 are *uniform bins*, not MOSI integers

A common MOSI protocol maps scores to the 7 integers
`{-3,-2,-1,0,1,2,3}` by rounding. **This repo does something else.**

`split_uniform_7` cuts `[-3, 3]` into **7 equal-width bins** of size
`6/7 ≈ 0.857`:

```
edges = [-3.0, -2.143, -1.286, -0.429, 0.429, 1.286, 2.143, 3.0]
```

`numpy.digitize(..., right=False)` then clips to `{1..7}`.
`split_uniform_5` is the same idea with 5 bins of width `1.2`.

| Bin | Acc-7 interval | Acc-5 interval |
| --- | --- | --- |
| 1 | [-3.000, -2.143) | [-3.0, -1.8) |
| 2 | [-2.143, -1.286) | [-1.8, -0.6) |
| 3 | [-1.286, -0.429) | [-0.6,  0.6) |
| 4 | [-0.429,  0.429) | [ 0.6,  1.8) |
| 5 | [ 0.429,  1.286) | [ 1.8,  3.0] |
| 6 | [ 1.286,  2.143) | — |
| 7 | [ 2.143,  3.000] | — |

`y == 3.0` lands in the last bin because of the clip. Values slightly
outside `[-3, 3]` (MOSEI can emit those) also clip.

**Do not compare Acc-7 / Acc-5 in this repo's CSVs to papers that
round to integers.** The uniform bins change the class prior — bin 4
is a narrow band around zero, not the whole `(-0.5, 0.5)` neutral
bucket.

`single_test` also draws a matplotlib confusion matrix of the Acc-7
bins (`plt.show()`). Headless example scripts skip that plot.

## What is written to CSV

Every `train_*.py` / `mosi_test/*.py` keeps:

```
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

MSE, raw test loss, and the confusion matrix are only printed.

## Trainer details that affect these numbers

- Objective during training: `torch.nn.L1Loss` (MAE), not MSE and not
  correlation.
- Optimizer: `AdamW`, `lr=1e-4`, `weight_decay=0.01`.
- Gradient clip: `clip_val=8`.
- Early stop: patience 7 on **validation L1**, if `early_stop=True`.
- Best checkpoint: lowest validation L1, saved with `torch.save(model, path)`
  (full module, not `state_dict`).
- `single_test` runs under `torch.no_grad()`; batch norm / dropout are
  switched off via `model.eval()`.

## Worked numerical example

`examples/metrics_walkthrough.py` uses a 12-point toy vector so you can
step through exclude-zero Acc-2, uniform Acc-7, and Pearson r by hand.
The same helpers are imported by `examples/run_all.py`.
