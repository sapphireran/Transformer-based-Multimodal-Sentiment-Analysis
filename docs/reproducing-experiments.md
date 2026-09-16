# Reproducing experiments

This is a checklist for someone who has (or can build) the four pickle
dumps. If you only want to see the modules run, use
[`examples/`](../examples/README.md) instead — those do not reproduce the
CSV numbers and are not claimed to.

## 0. Environment

```bash
python -m pip install -r requirements.txt
# GPU strongly recommended for TFN (128k-wide MLP) and GMTM (9 encoders)
```

Confirm:

```python
import torch
assert torch.cuda.is_available()
from model.models import GatedMultiTransfomerModel  # or cd model first
```

The original scripts assume `import models` works, which means **cwd =
`model/`** (they `sys.path.append(os.getcwd())`).

## 1. Features

Place:

```
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

To rebuild from MMSDK `.csd` files, use the notebooks under
`model/data/MOSEI/` and `model/data/MOSI/`, and the raw names in
`model/data/readme.md`. Alignment must be **word-synchronous** with
per-timestep BERT or GloVe, FACET 35, COVAREP 74, `T` long enough that
`Affectdataset` can cut / pad to 50.

Sanity peek (mirrors `analysis_dataset.ipynb`):

```python
import pickle
d = pickle.load(open("data/MOSEI/mosei_raw_bert.pkl", "rb"))
print(d.keys())                          # train, valid, test
print(d["train"]["vision"].shape[-1])    # 35
print(d["train"]["audio"].shape[-1])     # 74
print(d["train"]["text"].shape[-1])      # 768
```

## 2. MOSEI BERT fusion sweep

Edit `train_main_bert.py`:

- set `total_epochs` to something like `20` (the file currently has `1`)
- keep `early_stop=True`

```bash
cd model
python train_main_bert.py
```

Writes `checkpoints/{ConcatEarly,ConcatLate,LowRankTensorFusion,TensorFusion,TransformerEarly,TransformerLate}.pt`
and `main_results.csv`. Copy the CSV to `results/main_results.csv` if you
want the plotter to pick it up.

Expect: TransformerLate lowest MAE among the six; LMF second. Absolute
values will drift (no seed).

## 3. MOSEI GloVe fusion sweep

Uncomment `train()` in `train_main_glove.py`, then:

```bash
python train_main_glove.py
```

Watch `LateFusionTransformer(in_dim=1792)` vs actual concat width
`64+128+512=704`. If a fresh run crashes on `Conv1d`, set `in_dim=704`
(or whatever `sum(encoder embed dims)` is). The published
`glove_TransformerLate.pt` was produced by whatever width the script had
at that time.

## 4. GMTM + ablation

Uncomment `train()` in `train_GMTM_bert.py` / `train_GMTM_glove.py`.
Restore the seven-combination list in the BERT file. Then:

```bash
python train_GMTM_bert.py
python train_GMTM_glove.py
```

Checkpoints land in `checkpoints/ablation/`. Each subset is a full 3-stream
GMTM with zeros, so wall time is roughly 7× one full run, not 7 smaller
runs.

## 5. MOSI transfer

Needs the MOSEI `.pt` files from steps 2–4.

```bash
cd model/mosi_test
python train_mosi_bert.py
python mult_bert_mosi.py
```

GloVe transfer: `train_mosi_glove.py` currently imports MultiBench
(`training_structures.Supervised_Learning`). Either put MultiBench on
`PYTHONPATH` or change the import to `from train_and_test import test`
like the BERT script. `mult_glove_mosi.py` already uses `train_and_test`.

## 6. Compare to the checked-in CSVs

```bash
python examples/run_results_table.py
python examples/plot_published_results.py
```

A rerun is “close enough” if:

- BERT-MOSEI ranking is T-Late / LMF / TFN before the concat pair
- GMTM full MAE ≤ text-only MAE on MOSEI BERT
- MOSI MAE is in the 0.9–1.2 band, not the 0.5 band

Fourth-decimal identity is not expected.

## 7. Things that will silently give you a different paper

- Using uneven MOSI Acc-7 bins instead of `split_uniform_7`
- Forgetting `exclude_zero=True` on Acc-2 / F1
- Evaluating MOSI on the official test split only, while these scripts
  concat train+valid+test
- Loading a BERT checkpoint into a GloVe constructor
- Training GMTM with `DefaultHyperParams` (9-d) and comparing to the
  64-d CSV
- Reporting `train_main_bert.py` with `total_epochs=1` as a full run

## 8. What this documentation pass does *not* rerun

No `.pkl` files are in the workspace used to write these docs, so the
CSVs were **not** regenerated here. The CPU examples only prove that
fusion modules accept the documented shapes and that the metric helpers
match the formulas in `train_and_test.py`.
