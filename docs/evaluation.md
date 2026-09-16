# Evaluation

The official scoreboard is produced by `single_test` in
`model/train_and_test.py`. The same arithmetic is reimplemented without
a DataLoader or matplotlib in `examples/metrics.py` so you can unit-test
it on tensors.

The model always emits a **scalar** `ŷ ∈ ℝ` (trained with L1 against
the MOSI/MOSEI score). Classification numbers are derived from that
scalar; there is no separate classifier head.

## Regression metrics

Let `y` be the gold score and `ŷ` the prediction, both length `N`.

| Name | Formula | Notes |
| --- | --- | --- |
| Test loss | mean of the criterion (L1 in every script) | Same units as MAE when criterion is L1 |
| MAE | `mean(\|ŷ − y\|)` | Primary regression number in the CSVs |
| MSE | `mean((ŷ − y)²)` | Printed, not written to the CSVs |
| Corr | Pearson `r(y, ŷ)` via `scipy.stats.pearsonr` | Sensitive to a few outliers on MOSI |

`single_test` computes MAE / MSE in torch and Pearson in scipy after
flattening. If either vector is constant, Pearson is undefined; the
synthetic demo guards that case.

## Binary sentiment (Acc-2, F1)

`eval_affect(truths, results, exclude_zero=True)`:

1. Optionally drop indices where `y == 0` (`exclude_zero=True`, the
   default). Neutral clips are a large slice of MOSEI; dropping them
   is the usual "non-zero" protocol.
2. Map both `y` and `ŷ` to booleans with `> 0`.
3. Report sklearn `accuracy_score` and binary `f1_score`.

The sign mapping inside `single_test` that builds `pred ∈ {-1, 0, +1}`
is **not** what Acc-2 uses. Acc-2 / F1 reuse the raw regression tensor
and the `> 0` test above. Exact-zero predictions are counted as
negative once zeros in `y` have been dropped.

## Acc-7 and Acc-5 (uniform bins)

MOSI/MOSEI scores live on `[-3, 3]`. The helpers
`split_uniform_7` / `split_uniform_5` cut that interval into 7 or 5
**equal-width** bins and return 1-based indices.

```text
Acc-7 edges: -3 + k * (6/7)   for k = 0..7
Acc-5 edges: -3 + k * (6/5)   for k = 0..5
```

`np.digitize(..., right=False)` then `np.clip(..., 1, n)` so a value
of exactly `+3` lands in the last bin rather than overflowing.

This is **not** the "round to nearest integer, then 7-way accuracy"
protocol used in some Multimodal Transformer papers. Equal-width bins
do not line up with `{-3,-2,-1,0,1,2,3}`. If you compare these CSVs
to a paper that says "Acc-7", check which binning they used before
calling the numbers comparable.

Gold and prediction are binned independently, then scored with
`accuracy_score`. There is no ordinal penalty: predicting bin 1 when
the gold is bin 7 costs the same as being off by one.

## Confusion matrix

`single_test` builds a 7-way confusion matrix of `(pred_7, true_7)`
and calls `plt.show()`. That display is for interactive notebooks; it
is not saved. Headless runs should set `MPLBACKEND=Agg` or skip the
block. `examples/metrics.py` never opens a figure.

## What is stored in the CSVs

Header used by every writer:

```text
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Values are `round(..., 4)` Python floats. Ablation rows store the
modality list (`['text']`, `"['text', 'audio']"`, …) in the first
column rather than a fusion name.

`model/results/*.csv` is the copy of record for MOSEI. Files next to
the training scripts (`model/main_results.csv`, …) are duplicates from
the original working directory.

## Protocol differences that change the number

| Choice | Effect |
| --- | --- |
| Drop `y == 0` before Acc-2 | Raises Acc-2 / F1 on MOSEI (neutrals are easy to get wrong) |
| Equal-width vs integer Acc-7 | Can move Acc-7 by several points |
| MOSI loader concatenates train+valid+test | MOSI tables are **transfer-to-all-MOSI**, not the official test split |
| `exclude_zero` left at default | Matches the CSVs; set `False` only for a diagnostic |
| Evaluating a GPU checkpoint on CPU | Fine if you `map_location`; metrics are device-agnostic |

## Using the helper outside the train loop

```python
from examples.metrics import evaluate_regression

report = evaluate_regression(y_true, y_pred, exclude_zero=True)
print(report.as_dict())
```

`report` carries MAE, MSE, Corr, Acc-7, Acc-5, Acc-2, F1, plus the
binned integer arrays if you want to draw your own matrix. See
`examples/run_metrics_demo.py`.
