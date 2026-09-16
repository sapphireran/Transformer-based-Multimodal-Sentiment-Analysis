# Transformer-based Multimodal Sentiment Analysis

Personal research scratchpad for comparing fusion methods on **CMU-MOSEI** and **CMU-MOSI**.
The question I keep coming back to: once text is already strong, how much do audio and
vision actually buy, and which fusion story survives a domain shift from MOSEI to MOSI?

This repository is **personal experiment code**, not a product. Weights, raw `.csd` files,
and the large pickle caches are kept off git. What lives here is the training/eval loop,
fusion implementations, logged CSVs, and the lab notes in [`notes/`](notes/).

## What this repo studies

Three aligned modalities, one scalar sentiment target in `[-3, 3]`:

| Modality | Feature family | Dim in this repo |
| -------- | -------------- | ---------------- |
| Visual   | Facet 42 (processed) | 35 |
| Audio    | COVAREP | 74 |
| Text     | BERT last hidden, or GloVe 840B | 768 or 300 |

Two text fronts, same visual/audio:

- **BERT** (`mosei_raw_bert.pkl` / `mosi_raw_bert.pkl`)
- **GloVe** (`mosei_raw_glove.pkl` / `mosi_raw_glove.pkl`)

Two experiment families:

1. **Classic fusion bake-off** — early concat, late concat, tensor fusion, low-rank tensor fusion, early transformer, late transformer.
2. **GMTM** (`GatedMultiTransfomerModel`) — pairwise crossmodal transformers, softmax modality weights, per-modality gates, attention pooling. Used for the modality ablations.

Logged metrics: MAE, Pearson Corr, Acc-7 / Acc-5 (uniform bins on `[-3, 3]`), Acc-2 and F1 (nonzero polarity).

## Headline numbers I actually trust

These are copied from the CSVs under `model/` and `model/results/`. Full write-ups are in [`notes/`](notes/).

**MOSEI, BERT, six fusion methods** (`model/main_results.csv`):

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| ------ | ----- | ------- | ------- | ------ | ---- |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.8206 | 0.6788 | 0.8604 |
| **TransformerLate** | **0.5846** | **0.4675** | **0.8393** | **0.7041** | **0.8699** |

**MOSEI, GMTM BERT ablation** (`model/ablation_results.csv`): text alone is already `MAE 0.5687 / Acc-2 0.8404`. Adding audio+vision only moves MAE to `0.5640`. Audio-only and vision-only barely correlate with the label.

That gap is the whole project, in one sentence: **text carries the task; fusion is fighting over the leftover.**

## Repo map

```text
model/
  models.py                 fusion blocks + GMTM
  train_and_test.py         MultiFramework, train loop, eval
  train_main_bert.py        MOSEI BERT fusion bake-off
  train_main_glove.py       MOSEI GloVe fusion bake-off
  train_GMTM_bert.py        GMTM + BERT ablations
  train_GMTM_glove.py       GMTM + GloVe ablations
  data/get_dataloader.py    MOSI / MOSEI loaders, packed vs max-pad
  data/MOSEI/               alignment notebooks + mmsdk helper
  data/MOSI/
  mosi_test/                MOSEI→MOSI transfer eval
  results/                  copies of the CSVs + plot notebook
  checkpoints/              empty placeholders; weights stay local
notes/                      personal experiment notes (start here)
docs/                       architecture, metrics, setup, data pipeline
```

Longer file-by-file notes: [`docs/repo-map.md`](docs/repo-map.md).

## How I run things

From `model/`, after the pickles exist (see [`docs/setup.md`](docs/setup.md) and [`docs/data-pipeline.md`](docs/data-pipeline.md)):

```bash
# MOSEI fusion bake-off (BERT)
python train_main_bert.py

# MOSEI fusion bake-off (GloVe) — currently eval-only against saved ckpts
python train_main_glove.py

# GMTM ablations
python train_GMTM_bert.py
python train_GMTM_glove.py
```

MOSI scripts under `model/mosi_test/` load **MOSEI-trained** checkpoints and score MOSI. They are transfer tests, not MOSI-from-scratch training.

Caveats I keep repeating to myself:

- `train_main_bert.py` currently has `total_epochs=1`. The CSVs look like longer runs. Treat the script as a template, not a replay of the logged table.
- Several GMTM / GloVe `train(...)` calls are commented out and only `torch.load` + `test(...)` remain.
- `get_mosi_dataloader` concatenates train+valid+test before scoring. MOSI numbers are **not** a clean held-out split. Details in [`notes/04-mosi-transfer.md`](notes/04-mosi-transfer.md).

## Docs and lab notes

| Doc | What it is |
| --- | ---------- |
| [`docs/architecture.md`](docs/architecture.md) | Encoders, fusion blocks, GMTM forward pass |
| [`docs/metrics.md`](docs/metrics.md) | How Acc-7 / Acc-5 / Acc-2 / F1 / MAE / Corr are computed |
| [`docs/data-pipeline.md`](docs/data-pipeline.md) | Pickle layout, padding modes, ablation zeroing |
| [`docs/setup.md`](docs/setup.md) | Local env, expected files, what is *not* in git |
| [`notes/00-lab-index.md`](notes/00-lab-index.md) | Index of every experiment note |

## License

MIT. See [`LICENSE`](LICENSE). Datasets (CMU-MOSI / CMU-MOSEI, GloVe, BERT) keep their own licenses; I do not redistribute them here.
