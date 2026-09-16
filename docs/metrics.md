# Evaluation metrics

MOSI and MOSEI are scored as **regression** of a continuous sentiment in
`[-3, 3]`, then sliced into classification views. The helpers live in
`model/metrics.py` and are the only numbers the example scripts print.

## Regression

| Name | Definition |
| --- | --- |
| **MAE** | mean absolute error between prediction and label |
| **MSE** | mean squared error |
| **Corr** | Pearson correlation; reported as `0` when either side is constant |

Training uses `torch.nn.L1Loss` (MAE) with AdamW, `lr=1e-4`,
`weight_decay=0.01`, gradient clip `8`.

## Multiclass accuracy on uniform bins

`split_uniform_7` and `split_uniform_5` cut `[-3, 3]` into equal-width bins
and run `sklearn.metrics.accuracy_score` on the bin indices.

```
Acc-7 bins: width 6/7 ≈ 0.857
Acc-5 bins: width 6/5 = 1.2
```

Values outside `[-3, 3]` clip into the first / last bin. This is the
"uniform split" variant recorded in the CSVs as `Acc7_uniform` /
`Acc5_uniform`. It is **not** the human-anchored 7-class mapping used in
some earlier papers (those papers treat `±3, ±2, ±1, 0` as the labels
directly). Keep that in mind when comparing to tables that say "Acc-7"
without "uniform".

`examples/bin_edges.py` prints every edge. `examples/metric_sensitivity.py`
shifts otherwise-perfect predictions so you can see Acc-7 drop at a wall
while MAE only grows by the shift.

## Binary sentiment

`eval_affect`:

1. optionally drop clips whose label is exactly `0` (`exclude_zero=True`,
   the Multimodal Transformer convention)
2. threshold both truth and prediction at `0`
3. report binary accuracy (**Acc-2**) and binary F1 (**F1**)

## Shared schema

`compute_sentiment_metrics(truth, pred)` returns:

```python
{
    "MAE": float,
    "MSE": float,
    "Corr": float,
    "Acc7_uniform": float,
    "Acc5_uniform": float,
    "Acc2": float,
    "F1": float,
}
```

`train_and_test.single_test` adds `TestLoss` (the criterion averaged over
the loader) and can still draw a 7-class confusion matrix.

## Worked example

A toy clip with label `-1.2` and prediction `-0.9` contributes `0.3` to MAE.
Both fall in Acc-7 bin 3 and Acc-5 bin 2, and both are negative, so they
count as a binary true negative. Run `examples/evaluate_toy.py` to print a
full table on a synthetic set, or `pytest tests/test_metrics.py` to lock
the bin edges against the original `digitize` formula.
