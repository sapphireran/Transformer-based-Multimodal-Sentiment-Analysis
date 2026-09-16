# Evaluation protocol

Training minimizes **L1** on the continuous sentiment score
(`torch.nn.L1Loss`). Every other number in the CSVs is computed *after*
inference by [`single_test`](../model/train_and_test.py).

The implementation used by the examples is
[`examples/eval_protocol.py`](../examples/eval_protocol.py). It mirrors the
functions in `train_and_test.py` without pulling in `memory_profiler` or
`plt.show()`.

## Regression (what the model is trained for)

Let `y` be the gold score and `ŷ` the model output, both in (approximately)
`[-3, 3]`.

| Metric | Formula | Code |
| --- | --- | --- |
| MAE | `mean(\|ŷ − y\|)` | `torch.mean(torch.abs(...))` |
| MSE | `mean((ŷ − y)²)` | printed, not stored in the CSVs |
| Corr | Pearson `r(y, ŷ)` | `scipy.stats.pearsonr` |

`TestLoss` is the same L1 as MAE when the criterion is `L1Loss`, averaged
over the test loader.

## Binary sentiment (Acc2, F1)

`eval_affect(truths, results, exclude_zero=True)`:

1. Drop clips whose gold label is exactly `0` (neutral). This is the common
   MOSI/MOSEI “non-zero” setting.
2. Map the rest with `label > 0` → positive, else negative. The **same**
   threshold is applied to predictions.
3. Report sklearn `accuracy_score` (Acc2) and binary `f1_score`.

Neutral gold labels therefore never enter Acc2/F1. A model that dumps every
clip at `0.0` would score 0 on both after the `> 0` test.

## Fine-grained accuracy (Acc5, Acc7)

The continuous range `[-3, 3]` (width 6) is cut into **equal-width** bins.
This is **not** the “hashtag / integer-rounded” Acc7 used in some MOSI
papers. It is a uniform digitize:

```
# Acc7
step  = 6 / 7
edges = [-3 + i * step for i in range(8)]
bin   = clip(digitize(x, edges), 1, 7)

# Acc5
step  = 6 / 5
edges = [-3 + i * step for i in range(6)]
bin   = clip(digitize(x, edges), 1, 5)
```

`np.digitize(..., right=False)` is left-closed, right-open. Values of
exactly `+3` land past the last interior edge and are clipped into the
final bin.

Equal-width bins are uneven in *probability mass*: MOSI/MOSEI labels pile
up near 0, so Acc7 is dominated by the center bins. Compare methods with
the same protocol; do not drop these Acc7 numbers next to a paper that
used integer rounding (`round(y)` into `{-3,…,3}`).

Bin edges for Acc7, for reference:

| Bin | Interval (approx.) |
| ---: | --- |
| 1 | [−3.000, −2.143) |
| 2 | [−2.143, −1.286) |
| 3 | [−1.286, −0.429) |
| 4 | [−0.429, +0.429) |
| 5 | [+0.429, +1.286) |
| 6 | [+1.286, +2.143) |
| 7 | [+2.143, +3.000] |

Acc5 edges: `[-3.0, -1.8, -0.6, 0.6, 1.8, 3.0]`.

## What `single_test` also does

The original tester:

1. Runs the model in `eval()` / `no_grad`.
2. Collects raw `ŷ` for regression.
3. Builds a **sign** tensor (`+1 / −1 / 0`) that is **not** used for Acc2 —
   Acc2 uses the raw continuous `ŷ` inside `eval_affect`.
4. Draws a 7-class confusion matrix with `matplotlib` and calls `plt.show()`.
   Headless runs will hang or fail here. The example protocol skips the plot.

## Metric priority when reading tables

Use this order; it matches what the model optimized and what is most stable:

1. **MAE** and **Corr** — trained objective and ranking quality.
2. **Acc2 / F1** — standard binary MOSI/MOSEI non-zero split.
3. **Acc5 / Acc7** — only as a same-protocol comparison inside this repo.

A 0.004 MAE gap (GMTM text vs. text+audio+visual on MOSEI BERT) is small.
Treat it as “multimodal helps a little on this seed,” not as a claim that
needs a new paper title.
