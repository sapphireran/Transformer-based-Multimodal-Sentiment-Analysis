# Transformer-based Multimodal Sentiment Analysis

Personal research code for **multimodal sentiment analysis** on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
The project compares classical fusion baselines with transformer fusion, and
introduces a **Gated Multi-Transformer Model (GMTM)** that exchanges information
across text, audio, and vision with pairwise cross-modal attention, a learned
modality gate, and attention pooling over time.

This repository is personal work (MIT License, copyright pang990801 / Sapphire Ran).
It is not affiliated with a company codebase.

## What is in this repo

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, GMTM |
| [`model/train_and_test.py`](model/train_and_test.py) | `MultiFramework` trainer, MOSI/MOSEI metrics |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | Pickle loaders, padding, modality ablation |
| [`model/train_main_bert.py`](model/train_main_bert.py) / [`train_main_glove.py`](model/train_main_glove.py) | Baseline fusion sweep on MOSEI |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) / [`train_GMTM_glove.py`](model/train_GMTM_glove.py) | GMTM + modality ablations on MOSEI |
| [`model/mosi_test/`](model/mosi_test/) | Transfer / eval of MOSEI-trained weights on MOSI |
| [`model/results/`](model/results/) | Recorded metrics (MAE, Acc7/5/2, Corr, F1) |
| [`docs/`](docs/) | Architecture, datasets, metrics, experiment notes |
| [`examples/`](examples/) | CPU-friendly walkthroughs on **synthetic** tensors |

Recorded MOSEI BERT result (full text + audio + visual, GMTM):

| MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5640 | 0.4827 | 0.5561 | 0.8429 | 0.7255 | 0.8777 |

Late transformer fusion is the strongest *baseline* on MOSEI BERT (`MAE 0.5846`).
GMTM improves on that when all three modalities are present. Text carries most of
the signal; audio-only and vision-only runs stay near chance on the 7-class bins.

Full tables and interpretation: [`docs/experiments.md`](docs/experiments.md).

## Modalities and feature widths

Aligned sequences are truncated / padded to **50** steps.

| Modality | Source | Width |
| --- | --- | ---: |
| Vision | FACET 4.2 facial action features | 35 |
| Audio | COVAREP acoustic features | 74 |
| Text (BERT) | Contextual token embeddings | 768 |
| Text (GloVe) | `glove.840B.300d` word vectors | 300 |

Sentiment labels are continuous in roughly **[-3, 3]**. Training uses `L1Loss`
(MAE). Evaluation also reports Pearson correlation, binary accuracy / F1
(positive vs negative, zeros excluded), and uniform-bin Acc-5 / Acc-7.

## Architecture sketch

```
vision [B, T, 35] ─┐
audio  [B, T, 74] ─┼─► per-modality Linear + LayerNorm + ReLU
text   [B, T, D]  ─┘
                    │
                    ▼
         pairwise cross-modal TransformerEncoder (i ← j for all i, j)
                    │
                    ▼
         softmax modality weights + sigmoid gate
                    │
                    ▼
         concat over modalities → AttentionPooling over T
                    │
                    ▼
         LayerNorm → Linear → ReLU → Linear → scalar sentiment
```

See [`docs/architecture.md`](docs/architecture.md) for fusion baselines
(early/late concat, tensor fusion, low-rank tensor fusion, early/late
transformers) and the exact tensor shapes each module expects.

## Quick start (examples, no CMU downloads)

The examples do **not** need MOSI/MOSEI pickles or a GPU. They build synthetic
batches with the same ranks the real loaders emit, then run the modules in
`model/models.py`.

```bash
python -m pip install -r requirements.txt
python -m examples.run_all
python -m pytest tests/ -q
```

Individual walkthroughs:

```bash
python -m examples.forward_gmtm
python -m examples.fusion_shapes
python -m examples.metrics_walkthrough
python -m examples.tiny_train_loop
python -m examples.ablation_zeroing
python -m examples.results_tables
```

## Training on real data

1. Place processed pickles at `model/data/MOSEI/mosei_raw_bert.pkl` (or
   `mosei_raw_glove.pkl`) and the MOSI counterparts under `model/data/MOSI/`.
   Feature file names are listed in [`model/data/readme.md`](model/data/readme.md)
   and [`docs/datasets.md`](docs/datasets.md).
2. From `model/`:

```bash
cd model
python train_main_bert.py          # baseline fusion sweep, BERT features
python train_GMTM_bert.py          # GMTM (uncomment train(...) to fit)
```

Default optimizer in the scripts: **AdamW**, `lr=1e-4`, `weight_decay=0.01`,
batch size 32, gradient clip 8, early stop after 7 non-improving validation
epochs. Checkpoints are written under `model/checkpoints/` and
`model/checkpoints/ablation/`.

Reproduction notes, including MOSI transfer scripts: [`docs/reproduction.md`](docs/reproduction.md).

## Repository map

```
.
├── docs/                      # written notes for this personal project
├── examples/                  # synthetic, CPU-runnable walkthroughs
├── tests/                     # pytest coverage for examples + metric helpers
├── model/
│   ├── models.py              # GMTM and fusion implementations
│   ├── train_and_test.py      # MultiFramework + eval_affect
│   ├── train_*.py             # MOSEI experiment entry points
│   ├── data/                  # loaders + MMSDK notebooks
│   ├── mosi_test/             # MOSI evaluation of MOSEI-trained weights
│   ├── results/               # committed CSV tables
│   └── checkpoints/           # intended weight location (not in git)
└── LICENSE                    # MIT
```

## License

MIT. See [LICENSE](LICENSE).
