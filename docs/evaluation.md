# Evaluation

Training minimises L1 on the continuous MOSI / MOSEI score. Reporting
then adds classification numbers that the community uses around those
datasets. The helpers in `model/metrics.py` reimplement the same mapping
the training scripts use, without matplotlib.

## Continuous scores

Let `y` be the annotator mean in `[-3, 3]` and `ŷ` the model output.

| name | formula | notes |
| --- | --- | --- |
| MAE | mean `\|ŷ − y\|` | training loss |
| MSE | mean `(ŷ − y)²` | printed in `single_test`, not stored in the CSVs |
| Corr | Pearson `r(ŷ, y)` | undefined if either vector is constant |

`examples/04_metrics_walkthrough.py` compares an oracle copy of `y`, a
lightly noised copy, a heavily noised copy, and a sign flip so the
direction of each metric is obvious.

## Uniform Acc-7 and Acc-5

This repository does **not** use the CMU cut points `{-2.5, -1.5, ...}`.
It splits `[-3, 3]` into `n` equal-width bins and runs `numpy.digitize`
(left-closed), then clips the right endpoint so `y = 3` still lands in
the last bin.

Seven bins have width `6/7 ≈ 0.857`:

| label | interval |
| --- | --- |
| 1 | `[-3.000, -2.143)` |
| 2 | `[-2.143, -1.286)` |
| 3 | `[-1.286, -0.429)` |
| 4 | `[-0.429,  0.429)` |
| 5 | `[ 0.429,  1.286)` |
| 6 | `[ 1.286,  2.143)` |
| 7 | `[ 2.143,  3.000]` (clip) |

Five bins have width `1.2`. Acc-n is ordinary accuracy on those integer
labels. Because the bins are uniform, a model that hugs 0 (the mode of
MOSEI) can look strong on Acc-7 without being sharp on the tails.

`split_uniform_7` / `split_uniform_5` in `model/metrics.py` match the
helpers of the same name in `train_and_test.py`.

## Binary Acc-2 and F1

`eval_affect`:

1. optionally drop clips whose **true** score is exactly `0`;
2. map `score > 0` to the positive class for both truth and prediction;
3. report binary F1 and accuracy.

Neutral clips are common. Including them (`exclude_zero=False`) usually
lowers Acc-2 because 0 is not `> 0`. The CSV tables use the default
`exclude_zero=True`.

The training `single_test` also thresholds `ŷ` to `{-1, 0, +1}` when
building a stored classification tensor, but the F1 / Acc-2 numbers it
prints come from `eval_affect(true, pred_reg)` on the **raw** regression
output, not from that ±1 tensor.

## What the CSVs store

Header used across `model/results/` and `model/mosi_test/`:

```text
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

`model/metrics.row_for_csv` emits the same columns. Rounding is four
decimal places, matching the training scripts.

## Confusion matrices

`single_test` builds a 7-way confusion matrix with sklearn and calls
`plt.show()`. That blocks headless agents and CI. The examples never
open a figure. If you re-enable the plot locally, switch the backend to
`Agg` and `savefig` instead of `show`.

## Toy vs real numbers

A synthetic overfit (`examples/03_gated_transformer_toy.py`) only proves
that GMTM can drive MAE down on correlated Gaussian features. It is not
comparable to the MOSEI BERT GMTM row (MAE 0.5640 in
`model/results/ablation_results.csv`). Cite the CSV tables for dataset
numbers; cite the examples only as wiring tests.
