# Transformer-based Multimodal Sentiment Analysis

Personal research code for clip-level sentiment regression on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
The project compares classic multimodal fusion blocks against
transformer early/late fusion and a gated cross-modal transformer
(GMTM) that I implemented in `model/models.py`.

This repository is **personal / academic**. It does not contain
employer or company code. The original experiment scripts expect local
MOSI/MOSEI pickles that are not checked in. The `docs/` and `examples/`
trees added here are self-contained: they document the real training
setup and run on **synthetic tensors** with the same ranks as MOSI/MOSEI
so you can inspect the models on a laptop CPU.

## What is in this repo

| Path | Role |
| --- | --- |
| `model/models.py` | Encoders, fusion modules, and `GatedMultiTransfomerModel` |
| `model/train_and_test.py` | Supervised loop, `MultiFramework`, MOSI/MOSEI metrics |
| `model/train_main_bert.py` / `train_main_glove.py` | Fusion bake-off on MOSEI |
| `model/train_GMTM_bert.py` / `train_GMTM_glove.py` | GMTM + modality ablations on MOSEI |
| `model/mosi_test/` | Transfer / evaluation scripts on MOSI |
| `model/data/get_dataloader.py` | Pickle loaders, padding, ablation zeroing |
| `model/results/*.csv` | Published numbers from the runs above |
| `model/metrics.py` | Stand-alone metric helpers used by examples |
| `model/synthetic.py` | Fake MOSI/MOSEI-shaped batches |
| `docs/` | Architecture, data, training, metrics, results |
| `examples/` | Runnable CPU demos (no dataset download) |

## Quick start (synthetic examples, no dataset)

```bash
python -m pip install -r requirements-examples.txt
python examples/forward_fusion.py
python examples/train_toy_gmtm.py
python examples/metrics_demo.py
python examples/inspect_results.py
python -m pytest tests -q
```

These commands stay on CPU. They invent visual/audio/text tensors with
the Facet (35), COVAREP (74), and BERT (768) or GloVe (300) widths used
by the real scripts.

## Quick start (real MOSI / MOSEI)

1. Install the full stack: `python -m pip install -r requirements.txt`
   plus a CUDA PyTorch build if you have a GPU.
2. Place GloVe and the CMU SDK `.csd` files where
   [`docs/datasets.md`](docs/datasets.md) describes.
3. Build `mosei_raw_bert.pkl` / `mosei_raw_glove.pkl` (and the MOSI
   twins) with the notebooks under `model/data/`.
4. From `model/`, run one of the `train_*.py` entry points.

Checkpoints are written under `model/checkpoints/` and
`model/checkpoints/ablation/`. Those directories are storage locations
only; the `.pt` files themselves are gitignored.

## Fusion methods

Every bake-off script trains a `MultiFramework` of
`(encoders → fusion → head)` and reports MAE / Acc-7 / Acc-5 / Acc-2 /
Pearson / F1 on continuous labels in `[-3, 3]`.

| Method | Idea | Typical encoders |
| --- | --- | --- |
| ConcatEarly | Cat features on the last dim, then one sequence model | Identity + LSTM/MLP |
| ConcatLate | Sequence model per modality, then cat | LSTM / GRU |
| TensorFusion | Outer product with a trailing 1 (TFN) | GRUWithLinear |
| LowRankTensorFusion | Rank-constrained TFN | GRUWithLinear |
| TransformerEarly | Conv1d + 4-layer Transformer on the cat stream | Identity |
| TransformerLate | Per-modality `TransformerSeq`, then a late transformer | `TransformerSeq` |
| GMTM | Cross-modal transformers + gates + attention pooling | Identity (raw features) |

See [`docs/architecture.md`](docs/architecture.md) for tensor shapes and
[`docs/results.md`](docs/results.md) for the numbers in `model/results/`.

## Headline numbers (MOSEI, BERT features)

From `model/results/main_results.csv` and `model/ablation_results.csv`:

| Setup | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| GMTM text+audio+visual | **0.5640** | 0.4827 | **0.8429** | **0.7255** | **0.8777** |
| GMTM text only | 0.5687 | 0.4728 | 0.8404 | 0.7202 | 0.8741 |

Text dominates; audio and visual still help GMTM on MOSEI-BERT. On
MOSI transfer and on GloVe features the ranking shifts — details in
[`docs/results.md`](docs/results.md).

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — modules, shapes, GMTM
- [`docs/datasets.md`](docs/datasets.md) — MOSI/MOSEI files and loaders
- [`docs/training.md`](docs/training.md) — optimizers, scripts, flags
- [`docs/evaluation.md`](docs/evaluation.md) — metric definitions
- [`docs/results.md`](docs/results.md) — tables and a short read of them
- [`docs/reproducibility.md`](docs/reproducibility.md) — what you need to rerun
- [`docs/glossary.md`](docs/glossary.md) — MOSI/MOSEI jargon
- [`examples/README.md`](examples/README.md) — how to run the demos

## License

MIT. See [`LICENSE`](LICENSE). Copyright (c) 2024 pang990801.
