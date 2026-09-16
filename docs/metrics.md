# Metrics

Sentiment is trained as **scalar regression** with `torch.nn.L1Loss` (MAE).
After inference, `single_test` in `model/train_and_test.py` also reports
classification-style scores by bucketing the same continuous values.

The synthetic reimplementation used by the examples is
[`examples/lib/metrics.py`](../examples/lib/metrics.py). It follows the
formulas below and does not open a matplotlib window.

## Regression

Let `y` be the gold score and `ŷ` the model output, both flattened to
length `N`.

| Name | Formula in this repo | Notes |
| --- | --- | --- |
| Test loss | mean of the criterion (`L1Loss`) | same as MAE when criterion is L1 |
| MAE | `mean(\|y − ŷ\|)` | primary training objective |
| MSE | `mean((y − ŷ)²)` | reported, not used as the train loss |
| Corr | Pearson `r` via `scipy.stats.pearsonr` | `single_test` discards the p-value |

## Uniform Acc7

`split_uniform_7` partitions `[-3, 3]` into **seven equal-width bins**
(width `6/7 ≈ 0.857`) and maps each value to `{1, …, 7}` with
`numpy.digitize`, then clips.

Edges:

```
-3.000, -2.143, -1.286, -0.429, 0.429, 1.286, 2.143, 3.000
```

Acc7 is ordinary accuracy of `digitize(ŷ)` vs `digitize(y)`.

This is **not** the integer-rounding Acc7 used in some MOSI papers
(`round(clip(y, -3, 3)) + 3`). Uniform-width bins treat the interval as a
ruler; a prediction of `0.4` and `0.5` can land in different classes even
though both are near-neutral.

## Uniform Acc5

Same idea with five bins (width `6/5 = 1.2`):

```
-3.0, -1.8, -0.6, 0.6, 1.8, 3.0
```

## Binary Acc2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Drop indices where the gold label is exactly `0` (when `exclude_zero`).
2. Map remaining gold and prediction to booleans with `> 0`.
3. Report sklearn `accuracy_score` and binary `f1_score`.

Neutral gold labels therefore do not enter Acc2 / F1. Predictions of exact
`0.0` are counted as negative (`> 0` is false). The sign mapping used when
building the printed `pred` tensor (`+1 / −1 / 0`) is **not** what Acc2 / F1
use; those metrics compare the raw regression values to zero.

## Confusion matrix

`single_test` builds a 7-way confusion matrix of `pred_7` vs `true_7` and
calls `plt.show()`. That is why importing `test()` on a headless machine can
need an `Agg` backend. The examples never call `single_test`.

## How to read a row in the CSVs

Example from `model/results/main_results.csv`, TransformerLate on MOSEI BERT:

```
MAE 0.5846 | Acc7 0.4675 | Acc5 0.5460 | Acc2 0.8393 | Corr 0.7041 | F1 0.8699
```

Lower MAE is better. Higher Acc / Corr / F1 is better. Acc2 and F1 will
almost always look much stronger than Acc7 because they only ask for the
sign of sentiment, on non-neutral clips.

[`examples/metric_walkthrough.py`](../examples/metric_walkthrough.py) runs
the bin edges and a few hand-checked cases so the mapping is inspectable
without a GPU.
