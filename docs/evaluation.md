# Evaluation metrics

`single_test` in `model/train_and_test.py` always treats the model as a **regressor** that emits `[B, 1]`, even though `criterion` defaults to CrossEntropy in the function signature. Recorded calls pass `criterion=torch.nn.L1Loss()`.

The same metric functions are re-implemented (without matplotlib / a live model) in `examples/lib/metrics.py` so you can unit-test the binning math.

## Regression

Let `y` be the gold score and `ŷ` the prediction (both flattened).

| Name | Definition |
| --- | --- |
| Test loss | mean of `criterion(ŷ, y)` over the loader (L1 in recorded runs) |
| MAE | `mean(\|ŷ − y\|)` |
| MSE | `mean((ŷ − y)²)` |
| Corr | Pearson `r` via `scipy.stats.pearsonr` |

MOSI/MOSEI sentiment is conventionally in **`[-3, 3]`**. Predictions are not clipped before MAE / Corr.

## Binary Acc-2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Drop indices where `y == 0` (neutral), unless `exclude_zero=False`.
2. Map the rest with `(value > 0)` → `{False, True}`.
3. Report `sklearn` binary F1 and accuracy.

`single_test` also builds a discrete `{−1, 0, +1}` tensor from the sign of `ŷ`, but **Acc-2 / F1 use the raw continuous `ŷ`**, not that sign tensor. Zeros in `ŷ` would count as negative in the `(> 0)` test.

## Uniform Acc-5 and Acc-7

The code does **not** use the “official” MOSI 7-class edges `{-3,-2,-1,0,1,2,3}` as class ids. It slices `[-3, 3]` into **equal-width** bins.

```
# Acc-7
step  = 6 / 7
edges = [-3 + i * step for i in 0..7]   # 8 edges
idx   = clip(digitize(x, edges), 1, 7)

# Acc-5
step  = 6 / 5
edges = [-3 + i * step for i in 0..5]
idx   = clip(digitize(x, edges), 1, 5)
```

`numpy.digitize(..., right=False)` puts `x == −3` in bin 1 and values at/above `+3` into the last bin after the clip.

Accuracy is `accuracy_score(true_bins, pred_bins)` on those integer ids. This is a **personal convention** in this repo; do not drop these Acc-5/7 numbers next to papers that use integer rounding (`round(y) + 3`) without a footnote.

## Confusion matrix

`single_test` draws a matplotlib confusion matrix for the 7-bin labels and calls `plt.show()`. Headless example scripts skip that plot. If you revive the original tester on a server, switch the backend to `Agg` or comment out the display.

## What “good” looks like on MOSEI

From the recorded BERT tables ([results.md](results.md)):

- Strong text models sit around **MAE 0.56–0.62**, **Acc-2 ~0.81–0.84**, **Corr ~0.66–0.73**.
- Audio-only / visual-only GMTM ablations have **Corr ~0.11–0.23** and MAE ~0.82 — barely above a constant predictor on a skewed label distribution.

Use Acc-2 + MAE as the first sanity check; Acc-7 moves more slowly and is sensitive to the uniform edges.

## Worked numbers

`examples/metrics_walkthrough.py` builds a 12-point toy `y` / `ŷ` pair, prints every metric, and shows the 7-bin edges. `tests/test_metrics.py` locks the bin edges and the exclude-zero rule so a later edit to `examples/lib/metrics.py` cannot silently drift from this document.
