# Evaluation

Clip-level sentiment is treated as **regression** first. Classification
numbers are derived by thresholding or binning the same scalar.

The helpers in [`model/metrics.py`](../model/metrics.py) are a
CPU-friendly extract of `eval_affect`, `split_uniform_5`,
`split_uniform_7`, and the score block in `single_test`. Use them from
examples and tests; the training script still computes the same
formulas (plus a confusion-matrix popup).

## Regression

Let `y` be the gold score and `ŷ` the model output, both in (roughly)
`[-3, 3]`.

| Name | Definition |
| --- | --- |
| MAE | `mean(\|ŷ − y\|)` — also the training loss |
| MSE | `mean((ŷ − y)²)` |
| Corr | Pearson `r` between `y` and `ŷ` |

`single_test` reports MAE / MSE / Corr on the full test loader, zeros
included.

## Binary (Acc-2, F1)

`eval_affect`:

1. Optionally drop clips with `y == 0` (`exclude_zero=True`, the
   default, a.k.a. the non-zero protocol).
2. Map `y > 0 → 1`, else `0`. Same for `ŷ`.
3. Binary F1 (positive class) and accuracy.

Neutral clips are common on MOSEI, so Acc-2 / F1 are usually a bit
more optimistic than Acc-7.

## Acc-5 and Acc-7 (uniform bins)

The range `[-3, 3]` (span 6) is split into `k` equal-width bins.
`np.digitize` assigns 1-based indices; values of exactly `+3` clip
into the last bin.

```
k = 7 → width 6/7 ≈ 0.857
k = 5 → width 6/5 = 1.2
```

Acc-k is ordinary accuracy on those integer labels. This is **not**
the same as the official MOSI 7-class edges used in some papers
(those sometimes isolate 0 as its own class). The CSVs are labeled
`Acc7_uniform` / `Acc5_uniform` for that reason.

`split_uniform(data, n_bins)` generalizes the 5/7 helpers if you want
other granularities in a notebook.

## `single_test` extras

The original tester also:

- casts logits through a sign function solely to build `pred` (the
  regression path uses the raw `all_oute` vector instead)
- draws a 7-class confusion matrix with matplotlib (`plt.show()`)
- prints wall-clock inference time and parameter count

`examples/metrics_demo.py` skips the GUI and only prints the numeric
bundle.

## Reading a result row

A typical CSV line:

```
TransformerLate,0.5846,0.4675,0.5460,0.8393,0.7041,0.8699
```

| Column | Meaning |
| --- | --- |
| Fusion Method | Graph name |
| MAE | lower is better |
| ACC7 | uniform 7-class accuracy |
| Acc5 | uniform 5-class accuracy |
| ACC2 | non-zero binary accuracy |
| Corr | Pearson r |
| F1 | non-zero binary F1 |

When you compare two rows, trust MAE and Corr first — they use the
raw scores. Acc-7 is the harshest classification view because a 0.9
error already crosses a bin boundary.

## Toy-data caveat

Synthetic labels in `model/synthetic.py` are a tanh of a few text
coordinates plus noise. Metrics on that data are only a smoke test
("did the loop run, did MAE move?"). They are **not** comparable to
the MOSEI tables.
