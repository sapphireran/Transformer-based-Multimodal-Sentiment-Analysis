# Metrics

All numbers in the CSVs come from `train_and_test.single_test`. Training itself only
watches **validation L1 loss** (plus an unused `pts` list). Early stop patience is 7
epochs with no valid-loss improvement.

## Target

Labels are continuous sentiment in roughly `[-3, 3]`. The head emits a single scalar.
I train with `L1Loss`, so the optimization target **is** MAE, not MSE or BCE.

`eval_affect` and the Acc-2 path drop zeros when `exclude_zero=True` (the default).
Neutral utterances do not count toward Acc-2 / F1.

## Regression

Computed on the raw scalar predictions vs raw labels:

| Name | Definition in code |
| ---- | ------------------ |
| TestLoss | Mean of the criterion (`L1Loss`) over the test loader |
| MAE | `mean(\|y − ŷ\|)` — should match TestLoss when criterion is L1 |
| MSE | `mean((y − ŷ)²)` — printed, **not written to CSV** |
| Corr | `scipy.stats.pearsonr` on the flattened vectors |

## Acc-7 and Acc-5 (uniform bins)

`split_uniform_7` / `split_uniform_5` cut `[-3, 3]` into equal-width bins and run
`sklearn.metrics.accuracy_score` on the bin ids.

```text
Acc-7 edges: -3 + i * (6/7)   for i = 0..7
Acc-5 edges: -3 + i * (6/5)   for i = 0..5
```

This is **not** the integer-rounding Acc-7 used in a lot of MOSI/MOSEI papers
(`round(clip(y, -3, 3))` → 7 classes). My tables are therefore **not** drop-in
comparable to MulT / MISA / Self-MM numbers unless I re-score with their binning.

I kept uniform bins because I wanted every class to have the same width on the
original scale. The cost is that the crowded region around 0 is not given extra
resolution, and papers that use rounding will look better/worse for reasons that
are not about the model.

`np.digitize(..., right=False)` plus `clip(..., 1, 7)` maps exactly `3.0` into the
last bin. Fine.

## Acc-2 and F1

`eval_affect`:

1. Drop indices where the true label is 0 (unless `exclude_zero=False`).
2. Binarize both truth and prediction with `> 0`.
3. Binary F1 and accuracy.

Predictions that land exactly on 0 become class 0 in the earlier `prede` list but
`eval_affect` uses the raw regression output, not that list. The `prede` tensor is
built and then ignored for the printed metrics. Dead code.

## What is in each CSV column

Header as written by the train scripts:

```text
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Some files say `ACC5`, some say `Acc5`. Same column.

`Fusion Method` is either a fusion name (`TransformerLate`) or a Python list
string (`['text', 'audio']`). I did not normalize that.

## Confusion matrix

`single_test` always computes a 7-way confusion matrix on the uniform bins and
calls `plt.show()`. That is why a MOSI/MOSEI eval on a headless box can hang.
The figure is not saved.

## Complexity wrappers

`all_in_one_train` / `all_in_one_test` print wall time, peak RSS (`memory_profiler`),
and parameter count. Those numbers were never copied into the CSVs.

## Protocol differences I have to remember when comparing rows

| Comparison | Same protocol? | Why it can lie |
| ---------- | -------------- | -------------- |
| MOSEI BERT vs MOSEI GloVe fusion tables | Yes, same `single_test` | Different text dim / encoder widths |
| MOSEI GMTM ablation vs fusion bake-off | Same metrics, **different model and padding** | GMTM uses max-pad 50 and Identity encoders |
| MOSI transfer vs MOSEI in-domain | Same metric code, **different test set construction** | MOSI loader merges train+valid+test |
| My Acc-7 vs a paper Acc-7 | **No** | Uniform bins vs rounded integers |

When I quote a number in the notes, I mean “this CSV, this protocol,” not “SOTA table.”
