# Evaluation

All numbers in `model/results/*.csv` and `model/mosi_test/*.csv` come from `single_test` / `test` in `model/train_and_test.py`. The model always emits a **scalar sentiment** in (roughly) `[-3, 3]`. Classification metrics are computed by binning that scalar.

## Continuous metrics

Let `y` be gold scores and `ŷ` the model output (both flattened):

| Name | Definition | Better |
| --- | --- | --- |
| TestLoss | mean of the training criterion (`L1Loss` here) | ↓ |
| MAE | `mean(\|ŷ − y\|)` | ↓ |
| MSE | `mean((ŷ − y)²)` | ↓ |
| Corr | Pearson `r` via `scipy.stats.pearsonr` | ↑ |

These three are the ones that match the usual MOSI/MOSEI papers most closely. Acc-7 in the literature is often defined with **integer rounding** (`round(clip(y, -3, 3))`), not the equal-width bins below. This repo uses the equal-width variant on purpose — do not mix the two when comparing to MulT / TFN papers.

## Equal-width Acc-7 and Acc-5

`split_uniform_7` / `split_uniform_5` slice `[-3, 3]` into `K` bins of width `6/K`:

```text
edges_7 = [-3 + i * (6/7) for i in 0..7]
edges_5 = [-3 + i * (6/5) for i in 0..5]
index   = clip(digitize(score, edges), 1, K)
```

Accuracy is vanilla `sklearn.metrics.accuracy_score` on those integer indices. A prediction of `+2.9` and a gold of `+2.1` can land in different bins even though both are “positive / strongly positive”. That is why Acc-7 moves around more than MAE on the same checkpoint.

## Binary Acc-2 and F1

`eval_affect(truths, results, exclude_zero=True)`:

1. Optionally drop gold zeros (`exclude_zero=True` is the default, matching the usual “non-neutral” MOSI/MOSEI binary setup).
2. Map the rest with `score > 0` → positive class.
3. Report binary accuracy and binary F1 (`average='binary'`).

Predictions that land **exactly** on 0 are treated as the negative class in the `> 0` test. `single_test` also builds a 3-way `{-1,0,+1}` tensor for printing, but Acc-2 / F1 use the raw regression output, not that tensor.

## Confusion matrix

`single_test` builds a 7-way confusion matrix (`pred_7` vs `true_7`) and calls `plt.show()`. Headless runs should either comment that block or set `MPLBACKEND=Agg`. The [examples](../examples/metrics_demo.py) reimplement the numeric part without plotting.

## MOSI vs MOSEI protocols in this repo

| Protocol | Loader | Split | Used by |
| --- | --- | --- | --- |
| In-domain MOSEI | `get_dataloader` | official test | `train_main_*.py`, `train_GMTM_*.py` |
| MOSI transfer | `get_mosi_dataloader` | **train+valid+test merged** | `mosi_test/train_mosi_*.py` |
| MOSI transfer + ablation | `get_ablation_mosi_dataloader` | merged, unused modalities zeroed | `mosi_test/mult_*_mosi.py` |

A MOSI CSV row is therefore **not** a standard MOSI test-set number. It is “MOSEI checkpoint scored on every MOSI clip the pickle still contains after `drop_entry`”.

## Output dict

`single_test` returns:

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

The train scripts keep four decimals and write:

```text
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Ablation CSVs put a Python list in the first column (`['text']`, `['text', 'audio']`, …). `examples/summarize_results.py` normalizes those to `text`, `text+audio`, etc.

## How to score one checkpoint by hand

```python
import torch
from train_and_test import test
from data.get_dataloader import get_dataloader

_, _, testdata = get_dataloader(
    "data/MOSEI/mosei_raw_bert.pkl",
    batch_size=32,
    max_pad=True,          # match how the checkpoint was trained
    data_type="mosei",
    num_workers=0,
)
model = torch.load("checkpoints/ablation/model_text+audio+visual.pt")
print(test(model, testdata, is_packed=False, criterion=torch.nn.L1Loss()))
```

Use `is_packed=True` and `max_pad=False` for concat / tensor / late-transformer checkpoints.

## Worked numeric example

The metrics demo uses a tiny hand-built batch (no model):

```text
y  = [ 2.0, -1.0,  0.5, -2.5,  0.0]
ŷ  = [ 1.8, -0.8,  0.2, -2.0,  0.1]
```

- MAE = `mean(0.2, 0.2, 0.3, 0.5, 0.1) = 0.26`
- Binary (drop the gold-zero): signs all match → Acc-2 = 1.0
- Acc-7 depends on which equal-width bin each value falls into — see the printed table from `python examples/metrics_demo.py`
