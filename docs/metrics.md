# Metrics

Training minimizes **L1** (`torch.nn.L1Loss`) between a scalar prediction and
the sentiment label. Evaluation in `single_test` reports both regression and
classification views of that same scalar. The implementation is
[`eval_affect`](../model/train_and_test.py), [`split_uniform_7`](../model/train_and_test.py),
and [`split_uniform_5`](../model/train_and_test.py).

Labels and predictions live on the CMU convention **approximately `[-3, 3]`**.
The binning helpers assume that range even if a model predicts slightly
outside it (`np.clip` after `digitize`).

## Regression

| Name | Definition in this repo |
| --- | --- |
| Test loss | Mean of the criterion (`L1Loss`) over the loader |
| MAE | `mean(\|y − ŷ\|)` on the raw tensors (same as L1 when the criterion is L1) |
| MSE | `mean((y − ŷ)²)` — reported, not trained |
| Corr | Pearson `r` via `scipy.stats.pearsonr` on the flattened vectors |

Higher Corr and lower MAE/MSE are better.

## Acc-7 and Acc-5 (uniform bins)

The scripts do **not** use the classic MOSI integer rounding (`round(y)` into
`{-3,-2,-1,0,1,2,3}`). They slice `[-3, 3]` into equal-width bins and take
`np.digitize`.

Acc-7:

```text
width = 6 / 7 ≈ 0.857
edges = [-3.0, -2.143, -1.286, -0.429, 0.429, 1.286, 2.143, 3.0]
category = clip(digitize(x, edges, right=False), 1, 7)
```

Acc-5:

```text
width = 6 / 5 = 1.2
edges = [-3.0, -1.8, -0.6, 0.6, 1.8, 3.0]
category = clip(digitize(x, edges, right=False), 1, 5)
```

Accuracy is then `sklearn.metrics.accuracy_score` on those integer codes.
Because the bins are uniform, a score of `0.0` is **not** its own class in
Acc-7 (it falls in the middle bin that straddles zero). That differs from
papers that treat `{−3,…,3}` as seven rounded integers. When you compare
tables, check the binning.

[`examples/metrics_demo.py`](../examples/metrics_demo.py) prints a few
boundary values so you can see which bin each score lands in.

## Acc-2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Optionally drop labels that are exactly `0` (`exclude_zero=True`, the default)
2. Binarize the rest with `value > 0` → positive class
3. `f1_score(..., average='binary')`
4. `accuracy_score`

So Acc-2 / F1 are **non-neutral polarity** metrics. Neutrals (`y == 0`) do not
count. A prediction of exactly `0.0` is mapped to class `0` in the helper that
builds the `{-1,0,1}` tensor, but the F1 path uses the raw regression tensor
and the `> 0` threshold — two slightly different views. The numbers written to
CSV come from `eval_affect` (raw regression, exclude zeros).

## Confusion matrix

`single_test` also builds a 7-way confusion matrix of `pred_7` vs `true_7` and
calls `plt.show()`. Headless environments need a non-interactive backend or
they will hang. The examples never call `single_test`.

## What a “good” row looks like on MOSEI BERT

From [`model/results/`](../model/results/):

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (text+audio+visual) | 0.5640 | 0.4827 | 0.8429 | 0.7255 | 0.8777 |

On this split GMTM is the best recorded fusion. Text-only GMTM is already
`0.5687` MAE — the non-text streams help, but they do not carry the task
alone (`audio` MAE `0.8306`, Corr `0.1124`).

## Reproducing the arithmetic

```bash
python3 examples/metrics_demo.py
python3 -m pytest tests/test_metrics.py -q
```

Those tests pin the bin edges and a couple of polarity cases so a future edit
to `split_uniform_*` or `eval_affect` cannot silently change the protocol.
