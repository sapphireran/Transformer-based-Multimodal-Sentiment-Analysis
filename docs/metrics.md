# Metrics

Training minimizes **L1** (`torch.nn.L1Loss`) on the continuous sentiment
score. Everything else is computed in `single_test()`
([`model/train_and_test.py`](../model/train_and_test.py)) after a full pass
over the evaluation loader.

The example module [`examples/eval_metrics.py`](../examples/eval_metrics.py)
reimplements the same formulas **without** opening a matplotlib window, so
headless runs and tests stay usable.

## Regression

Let `y` be the gold score and `ŷ` the model output, both flattened to `[N]`.

| Name | Definition | Code |
| --- | --- | --- |
| **MAE** | mean \|ŷ − y\| | `torch.mean(torch.abs(...))` |
| **MSE** | mean (ŷ − y)² | `torch.mean((...) ** 2)` |
| **Corr** | Pearson r | `scipy.stats.pearsonr` |
| **TestLoss** | criterion averaged over the set | `L1Loss` in the scripts |

MAE is the number the early-stopping loop also tracks (as validation L1).

## Binary classification (Acc-2, F1)

`eval_affect(truths, results, exclude_zero=True)`:

1. Drop indices where `y == 0` when `exclude_zero` is true (the default).
2. Map the rest to `{False, True}` with `value > 0`.
3. `sklearn.metrics.accuracy_score` → **Acc-2**.
4. `f1_score(..., average='binary')` → **F1**.

So Acc-2 is *positive vs negative on non-neutral clips*, not a 3-way
negative / zero / positive score. Neutral clips still affect MAE / Corr /
Acc-5 / Acc-7.

`single_test` also builds a discrete `{−1, 0, +1}` tensor from the raw
prediction (`>0`, `==0`, `<0`) but does **not** use that tensor for Acc-2;
Acc-2 always goes through `eval_affect` on the continuous values.

## Uniform-bin Acc-5 and Acc-7

The sentiment range **[−3, 3]** is split into equal-width bins.
`np.digitize` is 1-based; values are then clipped so the last edge maps
into the last class.

### Acc-7 — `split_uniform_7`

- Step = `6 / 7 ≈ 0.8571`
- Edges: `−3 + i * step` for `i = 0..7`
- Labels ∈ `{1, 2, 3, 4, 5, 6, 7}`

### Acc-5 — `split_uniform_5`

- Step = `6 / 5 = 1.2`
- Edges: `−3 + i * step` for `i = 0..5`
- Labels ∈ `{1, 2, 3, 4, 5}`

Accuracy is `sklearn.metrics.accuracy_score` on the binned `ŷ` vs binned `y`.
These are **not** the non-uniform “has7 / has5” bins used in some MOSI papers
(those papers often treat `±2.5, ±1.5, ±0.5` as boundaries, or collapse
`|y| < 0.5` specially). Numbers in this repo are only comparable to other
runs that use the same uniform edges.

`single_test` also draws a 7-class confusion matrix with
`ConfusionMatrixDisplay` and `plt.show()`. Example code skips that display.

## What the CSVs store

```
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Values are `round(metric, 4)` as written by the training / MOSI scripts.

## Worked numeric example

A tiny walkthrough lives in
[`examples/metrics_walkthrough.py`](../examples/metrics_walkthrough.py):

- gold `y = [-2.4, -0.2, 0.0, 1.1, 2.8]`
- pred `ŷ = [-2.1, 0.4, 0.1, 0.8, 2.2]`

Expected qualitative behavior:

- MAE is the mean absolute gap (here 0.38).
- Acc-2 ignores the `0.0` gold clip, then checks sign agreement on the other
  four (the `−0.2` vs `0.4` pair is a binary error).
- Acc-7 / Acc-5 depend on which uniform bin each value falls into.

## Complexity helpers

`all_in_one_train` / `all_in_one_test` wrap the loop with
`memory_profiler.memory_usage` and print wall time plus parameter count.
They are enabled when `track_complexity=True` (the `train()` default).
The MOSI / GMTM *test* path calls `all_in_one_test` and then runs
`single_test` a **second** time to return the dict — inference time in the
log is only the first pass.
