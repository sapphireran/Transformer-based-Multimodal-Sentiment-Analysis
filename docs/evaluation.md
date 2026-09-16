# Evaluation protocol

All reported numbers come from [`single_test`](../model/train_and_test.py)
on a scalar head trained with `L1Loss` (MAE). The model does not emit
class logits. Classification scores are derived by binning or
thresholding the predicted real value.

## Regression

Let `y` be the gold score and `ŷ` the model output, both shape `(N, 1)`.

| Name | Definition |
| --- | --- |
| TestLoss | mean `criterion` (L1 on the recorded runs) |
| MAE | `mean(\|y − ŷ\|)` |
| MSE | `mean((y − ŷ)²)` |
| Corr | Pearson `r` via `scipy.stats.pearsonr` on the squeezed vectors |

## Uniform Acc7 and Acc5

`split_uniform_7` and `split_uniform_5` cut `[-3, 3]` into equal-width
bins and run `numpy.digitize`.

```
# 7 bins, width 6/7 ≈ 0.857
edges7 = [-3 + i * 6/7 for i in range(8)]   # 8 edges, 7 intervals
# 5 bins, width 1.2
edges5 = [-3 + i * 6/5 for i in range(6)]
```

`digitize(..., right=False)` plus `clip(..., 1, n)` maps every value
onto `{1, …, n}`. Acc7 / Acc5 are then ordinary
`sklearn.metrics.accuracy_score` on those integer labels.

This is **not** the uneven CMU-SDK binning that some papers use
(`[-3,-2), …`). Scores here are comparable to other runs in *this*
repo, not automatically to every published MOSI/MOSEI table.

A walkthrough with hand-picked scores is
[`examples/evaluate_metrics.py`](../examples/evaluate_metrics.py).

## Binary Acc2 and F1

`eval_affect` (default `exclude_zero=True`):

1. Drop indices where `y == 0`.
2. Map the rest to `{False, True}` with `> 0`.
3. Binary F1 and accuracy.

Neutral clips therefore do not count in Acc2 / F1. They still count in
MAE / Corr / Acc7 / Acc5.

## Sign histogram inside `single_test`

The function also builds a 3-way sign vector (`-1 / 0 / +1`) from `ŷ`
but never scores it. Only the raw `ŷ` (via `all_oute`) is used for the
metrics above.

## Confusion matrix side effect

`single_test` draws an Acc7 confusion matrix with `plt.show()`. That
blocks in a headless shell and is skipped by the synthetic examples,
which call `eval_affect` / `split_uniform_*` directly.

## What the training loop monitors

`train()` saves `torch.save(model, path)` whenever **validation L1**
improves. Early stop fires after 8 epochs with no improvement
(`patience > 7`) if `early_stop=True`.

There is no metric-based model selection (no "save best Acc2"). The
checkpoint is the best validation MAE.

`test()` currently calls the inner process **twice** — once under
`all_in_one_test` (for timing / param count) and once to return the
dict — so a full eval prints the tables twice. The CSV writers read
the returned dict.

## MOSI numbers are a transfer check

`model/mosi_test/*.py` load **MOSEI-trained** `*.pt` files and run
`single_test` on MOSI loaders. They do not fine-tune. Combined with
the merged-split MOSI loader (see [data-pipeline.md](data-pipeline.md)),
treat those CSVs as a stress test, not a standard MOSI leaderboard
row.

## Reproduction snippet

```python
from train_and_test import split_uniform_7, split_uniform_5, eval_affect
import numpy as np
from sklearn.metrics import accuracy_score

y = np.array([-2.4, -0.2, 0.0, 1.1, 2.8])
yhat = np.array([-2.1, 0.3, 0.1, 0.9, 2.2])
acc7 = accuracy_score(split_uniform_7(y), split_uniform_7(yhat))
acc5 = accuracy_score(split_uniform_5(y), split_uniform_5(yhat))
f1, acc2 = eval_affect(y, yhat, exclude_zero=True)
```
