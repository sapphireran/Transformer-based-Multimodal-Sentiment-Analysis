# Transformer-based Multimodal Sentiment Analysis

Personal research code for comparing **early**, **late**, **tensor**, and **transformer** fusion on CMU-MOSI and CMU-MOSEI. The main model is a gated cross-modal transformer (`GatedMultiTransfomerModel` in `model/models.py`) that mixes language, acoustic, and visual streams, then regresses a continuous sentiment score in `[-3, 3]`.

This repository is a personal study project. The original training scripts expect local pickle files that are **not** checked in (they are large and come from CMU-MultimodalSDK). The `docs/` and `examples/` trees added here explain the architecture and let you run the fusion / metric / toy-train path on **synthetic batches** without downloading MOSI or MOSEI.

## What is in this repo

| Path | Role |
| --- | --- |
| `model/models.py` | Encoders, fusion modules, gated multi-transformer (GMTM) |
| `model/train_and_test.py` | `MultiFramework` wrapper, train loop, MOSI/MOSEI metrics |
| `model/data/get_dataloader.py` | Pickle loaders, padding, modality ablation masks |
| `model/train_main_*.py` | Baseline fusion bake-off on MOSEI (BERT / GloVe text) |
| `model/train_GMTM_*.py` | GMTM + modality ablation on MOSEI |
| `model/mosi_test/` | Cross-dataset test of MOSEI-trained checkpoints on MOSI |
| `model/results/` | Recorded CSV tables from those runs |
| `docs/` | Architecture, datasets, training, metrics, results notes |
| `examples/` | Runnable synthetic-data walkthroughs (no SDK required) |

## Modalities and feature sizes

Aligned clips are truncated / padded to **50** time steps.

| Stream | Source | Dim (BERT text) | Dim (GloVe text) |
| --- | --- | --- | --- |
| Visual | FACET 4.2 | 35 | 35 |
| Acoustic | COVAREP | 74 | 74 |
| Language | BERT / GloVe 840B | 768 | 300 |

Sentiment labels are continuous. Evaluation reports MAE, Pearson correlation, binary accuracy / F1 (non-zero clips), and uniform-bin Acc-5 / Acc-7 on `[-3, 3]`. See [docs/evaluation.md](docs/evaluation.md).

## Results snapshot (MOSEI, BERT text)

From `model/results/main_results.csv` and `model/results/ablation_results.csv`:

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- |
| Concat early | 0.618 | 0.442 | 0.808 | 0.665 | 0.848 |
| Concat late | 0.615 | 0.451 | 0.808 | 0.664 | 0.846 |
| Low-rank tensor fusion | 0.597 | 0.459 | 0.831 | 0.693 | 0.864 |
| Tensor fusion | 0.602 | 0.445 | 0.824 | 0.678 | 0.863 |
| Transformer early | 0.606 | 0.451 | 0.821 | 0.679 | 0.860 |
| Transformer late | 0.585 | 0.468 | 0.839 | 0.704 | 0.870 |
| GMTM (text+audio+visual) | 0.564 | 0.483 | 0.843 | 0.726 | 0.878 |

Language dominates the ablation. Audio-only and visual-only stay near chance correlation; adding them to text still helps a little on MAE / Acc-2. Full tables and MOSI transfer numbers are in [docs/results.md](docs/results.md).

## Quick start (synthetic examples)

These do **not** need MOSI/MOSEI pickles or a GPU.

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python examples/summarize_recorded_results.py
python examples/metrics_walkthrough.py
python examples/fusion_forward_pass.py
python examples/gmtm_forward_pass.py
python examples/toy_train_loop.py
python examples/model_inventory.py
```

`examples/fusion_forward_pass.py` and `examples/gmtm_forward_pass.py` import the real modules in `model/models.py` and run a CPU forward pass on MOSI-shaped noise. `examples/toy_train_loop.py` fits a tiny GMTM for a few steps so you can see loss move without the full datasets.

## Reproducing the original experiments

1. Install [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) and download the MOSI / MOSEI computational sequences listed in [docs/datasets.md](docs/datasets.md).
2. Build `mosei_raw_bert.pkl`, `mosei_raw_glove.pkl`, `mosi_raw_bert.pkl`, and `mosi_raw_glove.pkl` with the notebooks under `model/data/`.
3. From `model/`, run `train_main_bert.py` or `train_GMTM_bert.py` (training calls are commented in some scripts; uncomment `train(...)` before a from-scratch run).
4. Checkpoints are written under `model/checkpoints/` (main bake-off) and `model/checkpoints/ablation/` (GMTM + modality drops).

Details: [docs/training.md](docs/training.md) and [docs/reproducing.md](docs/reproducing.md).

## Architecture in one paragraph

`MultiFramework` encodes each modality, fuses the three tensors, then applies a regression head. Baseline fusions are concat-early, concat-late, tensor fusion, low-rank tensor fusion, and transformer early/late. GMTM instead projects every stream to a shared `embed_dim`, runs a full cross-modal transformer grid (`i` attending to `j` for all pairs), softmax-weights those views, applies a per-modality sigmoid gate, concatenates, attention-pools over time, and maps to a scalar. See [docs/architecture.md](docs/architecture.md) and [docs/fusion-methods.md](docs/fusion-methods.md).

## License

MIT. Copyright (c) 2024 pang990801.
