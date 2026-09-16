# Transformer-based Multimodal Sentiment Analysis

Personal research code for **multimodal sentiment regression** on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
The project compares classical fusion baselines with transformer fusion, and
adds a gated cross-modal transformer (**GMTM**) used for modality ablations.

Inputs are aligned **visual**, **acoustic**, and **text** sequences. The
prediction target is a continuous sentiment score on roughly `[-3, 3]`.
Reported metrics include MAE, Pearson correlation, binary accuracy / F1, and
uniform 5-way / 7-way accuracy.

This repository is personal study and experiment code. It is not a company
product and is not packaged as a library.

## What is in this repo

| Area | Location | Role |
| --- | --- | --- |
| Model blocks and GMTM | [`model/models.py`](model/models.py) | Encoders, fusion layers, gated transformer |
| Train / test loop | [`model/train_and_test.py`](model/train_and_test.py) | `MultiFramework`, L1 training, metric suite |
| MOSEI fusion sweep | [`model/train_main_bert.py`](model/train_main_bert.py), [`model/train_main_glove.py`](model/train_main_glove.py) | Early / late concat, tensor fusion, transformers |
| GMTM + ablation | [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py), [`model/train_GMTM_glove.py`](model/train_GMTM_glove.py) | Zeroed-modality ablations |
| MOSI transfer tests | [`model/mosi_test/`](model/mosi_test) | Evaluate MOSEI-trained checkpoints on MOSI |
| Data loading | [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | Packed vs max-padded loaders |
| Recorded numbers | [`model/results/`](model/results) | CSV tables used in the docs |
| Written notes | [`docs/`](docs) | Architecture, data, training, metrics, results |
| Runnable toys | [`examples/`](examples) | CPU synthetic demos that do not need MOSI/MOSEI |

## Fusion methods compared

The main sweep builds a `MultiFramework(encoders, fusion, head)` for each
recipe. Feature widths below are **BERT-text** on MOSEI (`visual=35`,
`audio=74`, `text=768`). GloVe text is `300` instead of `768`; see
[`docs/architecture.md`](docs/architecture.md).

| Method | Encoders | Fusion | Head |
| --- | --- | --- | --- |
| `ConcatEarly` | identity | concat on feature dim (`877`) | packed LSTM → MLP |
| `ConcatLate` | per-modality LSTM | flatten + concat (`1344`) | MLP |
| `LowRankTensorFusion` | GRU + linear | rank-32 LMF → `256` | MLP |
| `TensorFusion` | GRU + linear | outer-product TFN | wide MLP |
| `TransformerEarly` | identity, max-padded | `EarlyFusionTransformer` | MLP |
| `TransformerLate` | `TransformerSeq` per modality | `LateFusionTransformer` | MLP |
| `GatedMultiTransfomer` | identity | pairwise cross-modal GMTM | identity (head is inside GMTM) |

On MOSEI + BERT, **TransformerLate** is the strongest fusion-sweep entry
(`MAE 0.5846`, `Acc2 0.8393`, `Corr 0.7041`). Full GMTM with all three
modalities is slightly better still (`MAE 0.5640`, `Corr 0.7255`). Tables and
commentary live in [`docs/results.md`](docs/results.md).

## Quick start (synthetic examples)

The recorded MOSI / MOSEI pickles and `.pt` checkpoints are **not** checked
in. You can still exercise the modules on random tensors:

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

That runner hits every toy script: synthetic batches, fusion shape traces,
encoder I/O, packed vs padded layouts, metric helpers, and a short GMTM
overfit on fake labels. Details: [`examples/README.md`](examples/README.md).

## Train on real data

1. Download CMU-MOSI / CMU-MOSEI computational sequences (Facet, COVAREP,
   words, labels) as listed in [`docs/datasets.md`](docs/datasets.md).
2. Build `mosei_raw_bert.pkl` / `mosei_raw_glove.pkl` (and the MOSI twins)
   with the notebooks under `model/data/`.
3. From `model/`:

```bash
python train_main_bert.py          # six fusion methods, BERT text
python train_GMTM_bert.py          # gated transformer + optional ablations
```

Training uses AdamW, `lr=1e-4`, `weight_decay=0.01`, and **L1** loss.
`TransformerEarly` uses max-padded batches (`max_pad=True`); the other fusion
methods use packed sequences (`is_packed=True`). See
[`docs/training.md`](docs/training.md).

## Evaluation

`single_test` reports:

- regression: **MAE**, **MSE**, Pearson **Corr**
- ordinal bins: **Acc7** and **Acc5** on uniform splits of `[-3, 3]`
- polarity: **Acc2** and binary **F1**, typically dropping exact zeros

Metric definitions and pitfalls are in
[`docs/evaluation.md`](docs/evaluation.md). A headless walkthrough is
`examples/metrics_demo.py`.

## Repository layout

```
.
├── docs/                  written notes (start at docs/README.md)
├── examples/              CPU demos; no dataset download required
├── model/
│   ├── models.py
│   ├── train_and_test.py
│   ├── train_main_*.py
│   ├── train_GMTM_*.py
│   ├── data/              loaders + dataset notes
│   ├── results/           CSV tables
│   ├── checkpoints/       expected .pt locations (files not committed)
│   └── mosi_test/         MOSI evaluation scripts
├── requirements.txt
└── LICENSE                MIT
```

## Documentation map

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Fusion methods](docs/fusion_methods.md)
- [Datasets and features](docs/datasets.md)
- [Training](docs/training.md)
- [Evaluation metrics](docs/evaluation.md)
- [Ablation study](docs/ablation.md)
- [Recorded results](docs/results.md)
- [Reproduction notes](docs/reproduction.md)
- [Glossary](docs/glossary.md)
- [Repository map](docs/repo_map.md)
- [Script reference](docs/script_reference.md)

## License

MIT. Original upload copyright is recorded in [`LICENSE`](LICENSE).
