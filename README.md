# Transformer-based Multimodal Sentiment Analysis

Personal research code for **gated multi-transformer fusion** on CMU-MOSI and CMU-MOSEI.
The repo compares GMTM against early/late concat, tensor fusion, low-rank tensor fusion,
and transformer fusion, using BERT or GloVe text features plus Facet vision and COVAREP audio.

This is a personal project (MIT). It is not company code.

## What is in this repo

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, and `GatedMultiTransfomerModel` |
| [`model/train_and_test.py`](model/train_and_test.py) | Supervised train/eval loop and MOSI/MOSEI metrics |
| [`model/metrics.py`](model/metrics.py) | Shared evaluation helpers (MAE, Corr, Acc7/5/2, F1) |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | Pickle loaders for aligned multimodal sequences |
| [`model/train_main_bert.py`](model/train_main_bert.py) | MOSEI BERT fusion baselines |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) | MOSEI BERT GMTM / ablation |
| [`model/mosi_test/`](model/mosi_test) | **MOSI transfer tests** from MOSEI-trained checkpoints |
| [`model/results/`](model/results) | Logged MOSEI tables |
| [`docs/`](docs) | Architecture, datasets, evaluation, and MOSI-next notes |
| [`examples/`](examples) | CPU-friendly synthetic MOSI walkthroughs |

Raw CMU SDK `.csd` files and trained `.pt` weights are **not** checked in.
Place processed pickles under `model/data/MOSI/` and `model/data/MOSEI/` as described in
[`docs/datasets.md`](docs/datasets.md).

## Feature layout

Each utterance is a length-`50` aligned sequence:

| Modality | Source | Dim (BERT setup) | Dim (GloVe setup) |
| --- | --- | --- | --- |
| Vision | FACET 4.2 | 35 | 35 |
| Audio | COVAREP | 74 | 74 |
| Text | BERT-base / GloVe 840B | 768 | 300 |
| Label | opinion / sentiment | scalar in about `[-3, 3]` | same |

Batch tensors after `max_pad=True` are `(batch, 50, dim)` per modality.

## Quick start (synthetic, no GPU, no official pickles)

```bash
python -m pip install -r requirements-examples.txt
python examples/01_synthetic_mosi_dataset.py
python examples/02_fusion_forward_pass.py
python examples/03_gmtm_tiny_train.py
python examples/04_metrics_walkthrough.py
python examples/05_results_tables.py
python -m pytest tests -q
```

These examples build a MOSI-shaped toy pickle, run every fusion module, train a tiny GMTM
for a few CPU steps, and reprint the checked-in experiment tables.

## Train / test on real MOSI or MOSEI

1. Download CMU-MOSI / CMU-MOSEI computational sequences (see [`docs/datasets.md`](docs/datasets.md)).
2. Build `mosi_raw_bert.pkl` / `mosi_raw_glove.pkl` (or the MOSEI equivalents) with the notebooks
   under `model/data/MOSI` and `model/data/MOSEI`.
3. From `model/`:

```bash
# MOSEI BERT baselines (writes checkpoints/<Fusion>.pt)
python train_main_bert.py

# MOSEI BERT GMTM ablation
python train_GMTM_bert.py

# MOSI transfer: evaluate a MOSEI checkpoint on MOSI pickles
cd mosi_test
python train_mosi_bert.py
python mult_bert_mosi.py
```

Scripts currently assume a CUDA device (`.cuda()`). For a CPU-only smoke test use the
`examples/` stack instead of the full trainers.

## Headline numbers (from checked-in CSVs)

**MOSEI + BERT** — `model/results/main_results.csv` and `model/results/ablation_results.csv`

- Best listed fusion baseline: **TransformerLate**, MAE `0.5846`, Acc2 `0.8393`, F1 `0.8699`
- GMTM (text+audio+visual): MAE `0.5640`, Acc2 `0.8429`, F1 `0.8777`

**MOSI transfer + BERT** — `model/mosi_test/mosi_bert_results.csv`

- Best listed fusion baseline: **TransformerLate**, MAE `0.8986`
- GMTM (text+audio+visual): MAE `0.9493`
- Ablation: **text+visual** is stronger than the full trio (MAE `0.9044`)

**MOSI transfer + GloVe** — `model/mosi_test/mosi_glove_results.csv`

- GMTM is the strongest listed model (MAE `0.9748`, Acc7 `0.3152`)

The MOSI transfer gap — GMTM wins on MOSEI BERT but loses to TransformerLate on MOSI BERT —
is the starting point for [`docs/mosi.md`](docs/mosi.md).

## Evaluation protocol

Regression on the raw score, then derived classification:

- **MAE / MSE / Pearson r** on the continuous label
- **Acc7 / Acc5**: uniform bins of `[-3, 3]`
- **Acc2 / F1**: sign of the score, **excluding zeros** (`exclude_zero=True`)

Details and edge cases: [`docs/evaluation.md`](docs/evaluation.md).

## License

MIT. See [`LICENSE`](LICENSE).
