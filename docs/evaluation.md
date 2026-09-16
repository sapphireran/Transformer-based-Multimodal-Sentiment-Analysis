# Evaluation

`single_test()` in `model/train_and_test.py` always reports the same
bundle. Classification numbers are **derived from the regression output**;
the network itself is never trained with CrossEntropy in the published
scripts.

```
{
  'TestLoss': L1 on the raw scores,
  'MSE': mean squared error,
  'MAE': mean absolute error,
  'Corr': Pearson r,
  'Acc7_uniform': accuracy on 7 uniform bins,
  'Acc5_uniform': accuracy on 5 uniform bins,
  'Acc2': binary accuracy, zeros dropped,
  'F1': binary F1, zeros dropped,
}
```

`examples/run_metrics.py` and `examples/metrics_lib.py` reimplement the
binning helpers so you can run them without importing `memory_profiler`
or triggering `plt.show()`.

## Regression metrics

Let `y` be the true score and `ŷ` the model output, both shape `(N,)`
after squeeze.

| Name | Definition in this repo |
| --- | --- |
| TestLoss | `L1Loss` used as `criterion` (same as MAE when criterion is L1) |
| MAE | `mean(\|y − ŷ\|)` |
| MSE | `mean((y − ŷ)²)` |
| Corr | `scipy.stats.pearsonr(y, ŷ)` |

## Uniform Acc-7 and Acc-5

This is the part that differs from some MOSI papers.

`split_uniform_7` partitions `[-3, 3]` into **seven equal-width bins**
of `6/7 ≈ 0.857`. `numpy.digitize` (left-closed) assigns indices 1…7;
values at/above `+3` clip to 7.

`split_uniform_5` does the same with five bins of width `1.2`.

Accuracy is ordinary exact-match on those integer ids. Both the prediction
and the truth are binned — a prediction of `0.1` and a truth of `0.8`
can land in different Acc-7 buckets even though both are weakly positive.

**These are not** the classic MOSI “Acc-7” edges that isolate `0` as its
own class and use ±0.5 / ±1.5 / ±2.5. If you compare to a paper, check
which edges they used. The CSV headers say `ACC7` / `ACC5`; the code
paths are `Acc7_uniform` / `Acc5_uniform`.

Worked edges for Acc-7:

```
[-3.000, -2.143, -1.286, -0.429, 0.429, 1.286, 2.143, 3.000]
```

A score of `0.0` falls in bin 4 (`[-0.429, 0.429)`).

## Binary Acc-2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Optionally drop indices where `y == 0` (`exclude_zero=True`, the
   default — this is the usual MOSI/MOSEI “non-neutral” binary setup).
2. Map remaining values with `> 0` → positive class.
3. `sklearn.metrics.accuracy_score` and `f1_score(..., average='binary')`.

`single_test` also builds a three-way `{-1, 0, +1}` copy of the
predictions (`>0 / <0 / ==0`) but does **not** score that copy; Acc-2 / F1
use the raw continuous `pred_reg` vs `true_vals` through `eval_affect`.

## Confusion matrix side effect

After computing Acc-7, `single_test` draws
`ConfusionMatrixDisplay` on the 7-class ids and calls `plt.show()`.
Headless CI will sit there or fail depending on the backend. Prefer
`examples/metrics_lib.evaluate_affect_batch` for automation.

## What the CSVs store

Training scripts round every metric to 4 decimals and write:

```
Fusion Method, MAE, ACC7, ACC5, ACC2, Corr, F1
```

Ablation scripts put a Python list (or a quoted list) in the first
column: `['text']` or `"['text', 'audio']"`.
`examples/results_lib.py` normalizes those cells into `text`,
`text+audio`, etc.

MOSI result files sometimes spell the late-concat row as `Concat`
instead of `ConcatLate`, and they include a `GatedMultiTransfomer` row
that the MOSEI main table does not (GMTM lives in the ablation CSV
there). The plotter accepts both names.

## Metric direction

| Metric | Better |
| --- | --- |
| MAE, MSE, TestLoss | lower |
| Corr, Acc-7, Acc-5, Acc-2, F1 | higher |

There is no official “one number.” The README headline table sorts by
MAE because that is the training loss. GMTM wins that column on
BERT-MOSEI; text-only GMTM is already close, which is the main ablation
story.

## Recomputing a CSV row from a checkpoint

```python
from train_and_test import test
import torch

model = torch.load("checkpoints/TransformerLate.pt").cuda()
# testdata from get_dataloader(..., max_pad=False) for this method
row = test(model, testdata, is_packed=True, criterion=torch.nn.L1Loss())
```

`test()` calls `single_test` **twice** (once inside `all_in_one_test`
for the timer, once for the return value) and will show the confusion
matrix twice. That is harmless but slow.
