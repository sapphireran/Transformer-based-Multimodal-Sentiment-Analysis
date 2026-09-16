# Transformer-based Multimodal Sentiment Analysis

Personal research code for **aligned text–audio–visual sentiment regression** on
[CMU-MOSI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
and [CMU-MOSEI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).
The project compares classical fusion baselines against transformer fusion and a
gated cross-modal transformer (`GatedMultiTransfomerModel`, GMTM).

This repository is personal study work (MIT, © 2024 pang990801 / Sapphire Ran).
It is not a company codebase.

## What this repo contains

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, GMTM |
| [`model/train_and_test.py`](model/train_and_test.py) | Supervised loop, affect metrics |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | MOSI / MOSEI pickle loaders |
| [`model/train_main_bert.py`](model/train_main_bert.py) | MOSEI BERT fusion sweep |
| [`model/train_main_glove.py`](model/train_main_glove.py) | MOSEI GloVe fusion sweep |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) | GMTM + BERT ablation |
| [`model/train_GMTM_glove.py`](model/train_GMTM_glove.py) | GMTM + GloVe ablation |
| [`model/mosi_test/`](model/mosi_test) | Transfer / eval of MOSEI checkpoints on MOSI |
| [`model/results/`](model/results) | Recorded MAE / Acc / Corr / F1 tables |
| [`docs/`](docs) | Architecture, data, training, and result notes |
| [`examples/`](examples) | CPU-friendly synthetic demos of the real modules |

## Modalities and feature sizes

Every clip is a **word-aligned** sequence of length `T` (padded to 50 in most
scripts):

| Stream | Source | Width |
| --- | --- | --- |
| Visual | Facet 4.2 facial action units | 35 |
| Acoustic | COVAREP | 74 |
| Text (BERT) | Contextual token embeddings | 768 |
| Text (GloVe) | `glove.840B.300d` | 300 |

Labels are continuous sentiment in **[-3, 3]**. Training uses `L1Loss` (MAE).
At test time the same scalar is also binned for Acc-7 / Acc-5 / Acc-2 / F1.

## Fusion methods

The `MultiFramework` wrapper in `train_and_test.py` is always
`encoders → fusion → head`:

1. **ConcatEarly** — stack features on the channel axis, then LSTM + MLP.
2. **ConcatLate** — per-modality LSTM, flatten, concatenate, MLP.
3. **TensorFusion** — outer-product fusion with a leading-1 homogenization.
4. **LowRankTensorFusion** — rank-`r` factorization of the same tensor product.
5. **TransformerEarly** — 1×1 conv into a shared transformer over time.
6. **TransformerLate** — per-modality `TransformerSeq`, then a late transformer.
7. **GMTM** — pairwise cross-modal transformers, softmax modality weights,
   per-modality gates, attention pooling, and a regression head.

See [`docs/fusion-methods.md`](docs/fusion-methods.md) for tensor shapes and
[`docs/architecture.md`](docs/architecture.md) for GMTM internals.

## Headline numbers (from checked-in CSVs)

MOSEI + BERT, full text/audio/visual:

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (all three) | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

GMTM’s ablation on the same split shows **text carrying most of the signal**;
audio-only and visual-only stay near 0.82 MAE. Full tables and MOSI transfer
numbers live in [`docs/experiments-and-results.md`](docs/experiments-and-results.md).

## Quick start (synthetic, no CMU downloads)

The large `.pkl` / `.csd` corpora are **not** in git. The examples under
`examples/` build MOSI-shaped tensors in memory and exercise the real modules
on CPU:

```bash
python -m pip install -r requirements.txt
python examples/01_synthetic_batch.py
python examples/02_fusion_forward.py
python examples/03_gmtm_forward.py
python examples/04_metrics_walkthrough.py
python examples/05_mini_training.py
```

Each script prints shapes, a few numeric checks, and (for the mini trainer) a
short MAE curve. Details: [`examples/README.md`](examples/README.md).

## Training on real MOSI / MOSEI

1. Install extras in `requirements.txt` (PyTorch, scikit-learn, `mmsdk` if you
   rebuild features from CMU `.csd` files).
2. Place processed pickles where the scripts expect them:
   - `model/data/MOSEI/mosei_raw_bert.pkl`
   - `model/data/MOSEI/mosei_raw_glove.pkl`
   - `model/data/MOSI/mosi_raw_bert.pkl`
   - `model/data/MOSI/mosi_raw_glove.pkl`
3. From `model/`:

```bash
python train_main_bert.py          # fusion sweep, BERT / MOSEI
python train_GMTM_bert.py          # gated transformer, BERT / MOSEI
```

GloVe and MOSI counterparts are the `*_glove.py` and `mosi_test/` scripts.
Checkpoint directories are documented in
[`model/checkpoints/readme.md`](model/checkpoints/readme.md).
Dataset rebuild notes: [`docs/datasets.md`](docs/datasets.md).

**Optimizer defaults used in the training scripts:** AdamW, `lr=1e-4`,
`weight_decay=0.01`, gradient clip 8, `L1Loss`, early stop on validation MAE
(patience 7). Several scripts currently load an existing `.pt` and only run
`test()` — uncomment the `train(...)` block to fit from scratch.

## Metrics

`single_test` reports:

- **MAE / MSE / Pearson r** on the raw [-3, 3] prediction
- **Acc-7 / Acc-5** by slicing [-3, 3] into equal-width bins
- **Acc-2 / binary F1** after dropping exact-zero labels and thresholding at 0

Worked examples: [`docs/training-and-evaluation.md`](docs/training-and-evaluation.md)
and `examples/04_metrics_walkthrough.py`.

## Layout of the notes

- [`docs/architecture.md`](docs/architecture.md) — GMTM and the encoder/head pattern
- [`docs/fusion-methods.md`](docs/fusion-methods.md) — every fusion module, with shapes
- [`docs/datasets.md`](docs/datasets.md) — MOSI / MOSEI files and the loader
- [`docs/training-and-evaluation.md`](docs/training-and-evaluation.md) — loop and metrics
- [`docs/experiments-and-results.md`](docs/experiments-and-results.md) — CSV commentary
- [`docs/reproduction.md`](docs/reproduction.md) — how to rerun or extend a sweep
- [`docs/code-map.md`](docs/code-map.md) — file-by-file index
- [`docs/glossary.md`](docs/glossary.md) — MSA / fusion vocabulary

## License

MIT. See [`LICENSE`](LICENSE).
