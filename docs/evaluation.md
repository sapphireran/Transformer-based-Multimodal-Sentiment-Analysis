# Evaluation metrics

`single_test` in `model/train_and_test.py` always treats the model as a
**regressor**. It collects `(B, 1)` predictions, then derives every reported
number from those scalars and the gold scores.

The function also opens a matplotlib confusion matrix (`plt.show()`). That
is fine in a notebook and blocking in a headless shell. Use
`examples/metrics_demo.py` when you want the same math without a GUI.

## Inputs

| Tensor | Meaning |
| --- | --- |
| `pred_reg` | model outputs, shape `(N, 1)` or `(N,)` |
| `true_vals` | gold sentiment, same shape |

`single_test` additionally builds a 3-way sign snapshot (`-1 / 0 / +1`) for
an intermediate `pred` tensor; Acc2 / F1 do **not** use that snapshot. They
go through `eval_affect(true_vals, pred_reg)`, which thresholds the **raw**
regression values.

## Regression

```text
MSE  = mean((y - ŷ)²)
MAE  = mean(|y - ŷ|)
Corr = Pearson r(y, ŷ)
```

Training minimizes MAE (`L1Loss`), so the validation checkpoint and the test
MAE are the same objective. Pearson `r` is computed with
`scipy.stats.pearsonr` on the squeezed vectors.

## Uniform Acc7 and Acc5

MOSI / MOSEI papers often report 7-class accuracy on rounded integer labels.
This repo uses a **uniform-width** binning of `[-3, 3]` instead of
`round()`.

`split_uniform_7`:

```text
step  = 6 / 7
edges = [-3, -3+step, ..., 3]          # 8 edges, 7 bins
index = clip(digitize(x, edges), 1, 7)
```

`split_uniform_5` is the same with `step = 6 / 5` and bins `1..5`.

Accuracy is `sklearn.metrics.accuracy_score` on those integer codes. A
prediction of `3.0` is clipped into the last bin.

**Why this is not the same as “Acc-7” in some papers.** Many baselines map
`y` to `{−3,−2,−1,0,1,2,3}` by rounding. Uniform bins have different
boundaries (bin width `≈ 0.857` for 7-way). Compare only to other rows in
*this* table, or re-bin if you need a paper-faithful number.

The confusion matrix in `single_test` is 7-way, using these uniform codes.

## Binary Acc2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Optionally drop indices where `truth == 0` (`exclude_zero=True` is the
   default).
2. Binarize the rest with `value > 0` → positive class.
3. Return binary F1 (`average='binary'`) and accuracy.

So Acc2 here is **non-negative vs negative on non-zero gold**, not
“positive vs negative including zeros as a third class.” Neutral utterances
do not count in the denominator when `exclude_zero=True`.

F1 is the positive-class F1 of that same boolean pair, not macro-F1.

## Test loss

`TestLoss` is the mean of the same criterion passed into `test()` (L1 in
every script). It should track MAE closely for `L1Loss`; they can differ
slightly if reduction or device copies disagree, but they are the same
objective.

## What `test()` prints extra

`test()` calls `all_in_one_test`, which times a first `single_test` pass
(including the blocking plot), then **runs `single_test` again** to return
the dict. That double evaluation is wasteful and will show the confusion
matrix twice. When automating, call `single_test` directly or use the
example metric helpers.

Returned dict keys:

```python
{
  "TestLoss": ...,
  "MSE": ...,
  "MAE": ...,
  "Corr": ...,
  "Acc7_uniform": ...,
  "Acc5_uniform": ...,
  "Acc2": ...,
  "F1": ...,
}
```

CSV writers persist `MAE, ACC7, Acc5, ACC2, Corr, F1` and drop MSE / test
loss.

## How to read a row

A **strong** MOSEI BERT row in this repo looks like:

- MAE near `0.56–0.60`
- Acc7 near `0.45–0.48`
- Acc2 near `0.83–0.84`
- Corr near `0.70–0.73`
- F1 near `0.86–0.88`

Audio-only or visual-only GMTM rows sit around MAE `0.82` and Corr
`0.11–0.23` — the regression is barely better than predicting the mean, and
Acc2 near `0.63` is close to the positive-class prior after zeros are
dropped.

## Implementation notes for personal forks

- Keep `exclude_zero` consistent when you compare Acc2 across runs.
- If you switch training to `MSELoss`, still report MAE; it is the number
  the rest of the tables use.
- Pearson `r` is undefined if all predictions are constant. A collapsed
  model will throw inside `pearsonr` rather than return 0.
- `split_uniform_*` expects array-likes; passing a CUDA tensor works after
  `.cpu()` in `single_test`, but the helpers in
  `examples/metrics_demo.py` accept NumPy or tensors.

## Worked numbers

`examples/metrics_demo.py` uses a tiny hand-made `y` / `ŷ` pair and prints
each metric plus the bin edges. Use that script as a specification if you
reimplement the CSV columns in another language.
