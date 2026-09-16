# Evaluation protocol

Sentiment is trained as **L1 regression** on a scalar in roughly `[-3, 3]`.
Classification numbers are derived from that scalar after the fact.
The shared implementations live in [`model/metrics.py`](../model/metrics.py)
and are called from [`model/train_and_test.py`](../model/train_and_test.py).

## Training objective

`train(..., objective=torch.nn.L1Loss())` — mean absolute error between the
head output `(B, 1)` and the label. Checkpoints are saved on **validation L1**,
with optional early stop after 8 non-improving epochs (`patience > 7`).

## Regression metrics

On the test set, with `y` the gold score and `ŷ` the model score:

| Name | Definition |
| --- | --- |
| TestLoss | Mean of the same criterion used at train time (L1 here) |
| MAE | `mean(|y − ŷ|)` |
| MSE | `mean((y − ŷ)²)` |
| Corr | Pearson `r` between flattened `y` and `ŷ` (`scipy.stats.pearsonr`) |

These use **all** test points, including zeros.

## Acc7 and Acc5 (uniform bins)

`split_uniform_7` / `split_uniform_5` cut `[-3, 3]` into equal-width bins and
return 1-based indices.

```
Acc7 edges: -3 + i * (6/7)   for i = 0..7
Acc5 edges: -3 + i * (6/5)   for i = 0..5
```

`numpy.digitize(..., right=False)` then `clip` to `{1..K}`.
A prediction of exactly `3.0` lands in the last bin.

This is **not** the integer-round Acc7 used in some MOSI papers
(`round(clip(y, -3, 3)) + 3`). Numbers in this repo are only comparable to
other runs that used the same uniform edges. See
`examples/04_metrics_walkthrough.py` for a side-by-side on a toy vector.

## Acc2 and F1 (exclude-zero)

`eval_affect(truth, pred, exclude_zero=True)`:

1. Drop indices where `y == 0` (neutral utterances).
2. Map the rest to `{False, True}` with `y > 0` / `ŷ > 0`.
3. Binary F1 (`average='binary'`) and accuracy.

A model that always predicts a tiny positive value will look strong on a
positive-heavy split and silent on neutrals. MOSI transfer tables should be
read with that bias in mind.

`single_test` also builds a 7-way confusion matrix and calls `plt.show()`.
Headless example scripts use `compute_affect_metrics(..., plot_confusion=False)`.

## What `test()` returns

```python
{
    "TestLoss": tensor or float,
    "MSE": float,
    "MAE": float,
    "Corr": float,
    "Acc7_uniform": float,
    "Acc5_uniform": float,
    "Acc2": float,
    "F1": float,
}
```

CSV writers in the trainers persist
`Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1`
and drop TestLoss / MSE.

## MOSI transfer vs. MOSI test

`get_mosi_dataloader` **merges train+valid+test** before scoring.
Acc2/F1/MAE on those CSVs are therefore **not** official MOSI test-fold numbers.
They answer: "how does a MOSEI checkpoint behave on every MOSI segment we have?"

When you train on MOSI itself, score only `get_dataloader(...).test` and say so
in the table caption.

## Reproducing a metric-only check

```bash
python examples/04_metrics_walkthrough.py
python -m pytest tests/test_metrics.py -q
```

No GPU and no pickles required.
