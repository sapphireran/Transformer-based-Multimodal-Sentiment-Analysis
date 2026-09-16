# Transformer-based Multimodal Sentiment Analysis

Personal research code for **multimodal sentiment regression** on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/). The
repository compares classical fusion baselines against a gated pairwise
cross-modal Transformer (GMTM) using aligned visual, acoustic, and language
features.

Sentiment is treated as a continuous score on `[-3, 3]`. Training uses L1
loss. Evaluation reports both regression metrics (MAE, Pearson correlation)
and classification metrics obtained by thresholding or binning that score
(binary accuracy / F1, 5-class, 7-class).

This is a personal project. It is not affiliated with any employer.

## Why this repo exists

Most published multimodal affect models differ in two places: how each
modality is encoded, and how those encodings are fused. This codebase keeps
those two pieces swappable.

- **Encoders** can be identity (pass-through), LSTM, GRU, or a per-modality
  Transformer.
- **Fusion** can be early concatenation, late concatenation, tensor fusion,
  low-rank tensor fusion, early/late Transformers, or GMTM.
- **Heads** are small MLPs that map the fused vector to a scalar sentiment.

The same training loop in `model/train_and_test.py` therefore runs every
experiment. That makes it straightforward to compare fusion methods on the
same data, optimizer, and metrics.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── docs/                      # architecture, data, training, results
├── examples/                  # CPU-friendly synthetic demos
└── model/
    ├── models.py              # encoders, fusion modules, GMTM
    ├── train_and_test.py      # MultiFramework + train / test loop
    ├── train_main_bert.py     # MOSEI BERT fusion sweep
    ├── train_main_glove.py    # MOSEI GloVe fusion sweep
    ├── train_GMTM_bert.py     # MOSEI GMTM (BERT text)
    ├── train_GMTM_glove.py    # MOSEI GMTM (GloVe text)
    ├── data/                  # dataloaders + dataset notes
    ├── mosi_test/             # MOSI transfer / held-out eval
    ├── results/               # recorded CSV tables
    └── checkpoints/           # local weight dump (not in git)
```

Start with the docs if you want the design, or with `examples/` if you want
to run something immediately:

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Encoders, GMTM, MultiFramework |
| [docs/fusion-methods.md](docs/fusion-methods.md) | Every fusion operator and its tensor shapes |
| [docs/datasets.md](docs/datasets.md) | MOSI / MOSEI features, pickle layout, preprocessing |
| [docs/training.md](docs/training.md) | Scripts, hyperparameters, checkpoint paths |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions and binning rules |
| [docs/results.md](docs/results.md) | Recorded MOSEI / MOSI tables with notes |
| [docs/reproduction.md](docs/reproduction.md) | End-to-end reproduction checklist |
| [examples/README.md](examples/README.md) | Runnable synthetic walkthroughs |

## Feature dimensions

Aligned clips are truncated or padded to **50** timesteps.

| Modality | Source | Dimension |
| --- | --- | --- |
| Visual | FACET 4.2 action units | 35 |
| Acoustic | COVAREP | 74 |
| Language (BERT) | contextual word embeddings | 768 |
| Language (GloVe) | `glove.840B.300d` | 300 |

A BERT clip therefore has concatenated early-fusion width `35 + 74 + 768 = 877`.
A GloVe clip has width `35 + 74 + 300 = 409`.

## Models at a glance

### GMTM (Gated Multi-Transformer)

`GatedMultiTransfomerModel` is the main proposed model:

1. Project each modality to a shared `embed_dim`.
2. Run a full pairwise cross-modal Transformer grid: modality `i` attends to
   every modality `j` (including itself).
3. Softmax-weight those `n` attended streams and apply a learned sigmoid gate.
4. Concatenate the gated streams, pool time with attention, and predict a
   scalar.

Default experimental hyperparameters (see `train_GMTM_bert.py`) are 4 layers,
4 heads, `embed_dim=64`, and modality-specific attention dropout
`[0, 0, 0.1]` for visual / audio / text.

### Fusion baselines

Implemented in `model/models.py` and wired up in `train_main_*.py`:

| Name | Idea |
| --- | --- |
| `ConcatEarly` | Concatenate raw features on the last dim, then LSTM + MLP |
| `ConcatLate` | Encode each modality with LSTM, flatten, concatenate |
| `TensorFusion` | Outer-product fusion with a homogeneous bias 1 |
| `LowRankTensorFusion` | Rank-`r` factorization of the same tensor product |
| `TransformerEarly` | 1x1 conv + Transformer over the concatenated sequence |
| `TransformerLate` | Per-modality `TransformerSeq`, then a late Transformer |

## Recorded headline results (MOSEI, BERT)

From `model/results/main_results.csv` and `model/results/ablation_results.csv`.
Lower MAE is better; higher Acc / Corr / F1 is better.

| Method | MAE | Acc-7 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (text+audio+visual) | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

On this BERT-MOSEI sweep, GMTM is the strongest model and text is the
dominant modality. Audio-only and visual-only GMTM runs barely beat a
majority-ish baseline (MAE ≈ 0.82). See [docs/results.md](docs/results.md)
for GloVe and MOSI tables plus ablation commentary.

## Quick start (synthetic, no dataset download)

The examples do **not** need MOSI / MOSEI pickles or a GPU. They build
tiny batches with the same feature widths as the real loaders and run the
actual modules from `model/models.py`.

```bash
python -m pip install -r requirements.txt
python examples/run_fusion_demo.py
python examples/run_gmtm_demo.py
python examples/run_metrics_demo.py
python examples/run_tiny_train.py
```

Expected output is a set of tensor shapes, a GMTM forward pass, metric
numbers on a synthetic label vector, and a few SGD steps on fake data.

## Training on real MOSI / MOSEI

1. Download the CMU computational sequences listed in
   [docs/datasets.md](docs/datasets.md) and place them under
   `model/data/MOSI/cmumosi` and `model/data/MOSEI/cmumosei`.
2. Build `mosei_raw_bert.pkl` / `mosei_raw_glove.pkl` (and the MOSI
   equivalents) with the notebooks in `model/data/`.
3. From `model/`:

```bash
python train_main_bert.py          # six fusion baselines, BERT
python train_GMTM_bert.py          # GMTM, BERT
python train_main_glove.py         # six fusion baselines, GloVe
python train_GMTM_glove.py         # GMTM, GloVe
```

Checkpoints are written to `model/checkpoints/` and
`model/checkpoints/ablation/`. MOSI transfer scripts live in
`model/mosi_test/`.

**Note:** several training scripts currently load an existing `.pt` file
instead of calling `train(...)` (the train call is commented out). That is
intentional for re-evaluating recorded weights. Uncomment the `train(...)`
block to run a fresh fit. Details are in [docs/training.md](docs/training.md).

## Evaluation protocol

`single_test` in `model/train_and_test.py` reports:

- **MAE / MSE / Pearson r** on the raw `[-3, 3]` prediction
- **Acc-7 / Acc-5** by slicing `[-3, 3]` into 7 or 5 equal-width bins
- **Acc-2 / F1** by signing the score (`> 0` vs `< 0`), optionally
  dropping exact zeros

The same helpers are re-exported in `examples/metrics.py` so you can compute
them without constructing a full `DataLoader`. See
[docs/evaluation.md](docs/evaluation.md).

## Environment

- Python 3.10+ (examples were checked on 3.12)
- PyTorch 2.x (CPU is enough for `examples/`; CUDA is expected for the
  full MOSEI sweeps)
- A machine with enough RAM to hold the aligned pickle (MOSEI BERT is the
  largest)

## License

MIT. See [LICENSE](LICENSE).
