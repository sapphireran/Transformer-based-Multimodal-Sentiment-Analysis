# Transformer-based Multimodal Sentiment Analysis

Personal research code for **multimodal sentiment regression** on
[CMU-MOSI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
and [CMU-MOSEI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/).
The project compares classical fusion (early/late concat, tensor fusion, low-rank
tensor fusion) with transformer fusion and a gated cross-modal transformer
(GMTM).

This repository is a **personal** study notebook and experiment log, not a
packaged library. Scripts assume you already have aligned pickle features on
disk. The `examples/` folder is self-contained and runs on synthetic tensors
so you can inspect shapes, fusion math, and the evaluation protocol without
downloading MOSI/MOSEI.

## What is in here

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, GMTM |
| [`model/train_and_test.py`](model/train_and_test.py) | `MultiFramework` trainer + MOSI-style metrics |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | MOSI/MOSEI pickle loaders and ablation masks |
| [`model/train_main_bert.py`](model/train_main_bert.py) | Main fusion sweep (BERT text) |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) | Gated multi-transformer + modality ablation |
| [`model/results/`](model/results/) | Logged CSV tables from the original runs |
| [`docs/`](docs/) | Architecture, data, metrics, experiment notes |
| [`examples/`](examples/) | CPU-friendly synthetic demos |

A file-by-file map lives in [`docs/repo-map.md`](docs/repo-map.md).

## Modalities and feature sizes

Inputs are word-aligned sequences of length `T = 50`:

| Modality | Source | Dim |
| --- | --- | --- |
| Visual | FACET 4.2 | 35 |
| Acoustic | COVAREP | 74 |
| Text (BERT) | last-layer word pieces / aligned BERT | 768 |
| Text (GloVe) | `glove.840B.300d` | 300 |

The regression target is the continuous sentiment score on `[-3, 3]`.
Classification numbers (Acc2 / Acc5 / Acc7 / F1) are **derived** from that
score; they are not a separate training objective. See
[`docs/metrics.md`](docs/metrics.md).

## Fusion methods

```
vision  ──► encoder ──┐
audio   ──► encoder ──┼──► fusion ──► head ──► ŷ ∈ ℝ
text    ──► encoder ──┘
```

| Name | Idea | Typical encoders |
| --- | --- | --- |
| `ConcatEarly` | Concatenate along the feature axis, then one sequence model | Identity + LSTM |
| `ConcatLate` | Encode each modality, flatten, concatenate | LSTM / LSTM / LSTM |
| `TensorFusion` | Outer-product fusion with a constant-1 bias dim | GRU + linear |
| `LowRankTensorFusion` | Rank-`r` factorization of the same tensor | GRU + linear |
| `TransformerEarly` | Concatenate features, 1×1 conv, transformer | Identity |
| `TransformerLate` | Per-modality transformer, concat, second transformer | `TransformerSeq` |
| `GatedMultiTransfomerModel` | Pairwise cross-modal attention, learned gates, attention pooling | Identity (GMTM is the fusion) |

Architecture notes and tensor shapes: [`docs/architecture.md`](docs/architecture.md).

## Headline results (CMU-MOSEI, BERT text)

Numbers are copied from [`model/results/main_results.csv`](model/results/main_results.csv)
and [`model/results/ablation_results.csv`](model/results/ablation_results.csv).
Lower MAE is better; higher Acc / Corr / F1 is better.

**Main fusion sweep**

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | 0.8393 | 0.7041 | 0.8699 |

**GMTM modality ablation** (same BERT features)

| Modalities | MAE | Acc7 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.6293 | 0.2061 | 0.7693 |
| text+audio+visual | **0.5640** | 0.4827 | **0.8429** | **0.7255** | **0.8777** |

Takeaways that the ablation actually supports:

1. Text dominates. Audio or visual alone is close to a weak majority baseline.
2. Adding audio and visual to text still helps, but the gain is small
   (`0.5687 → 0.5640` MAE).
3. Among the non-GMTM fusion heads, late transformer fusion is the strongest
   on this BERT MOSEI split.

GloVe and MOSI-transfer tables are in [`docs/experiments.md`](docs/experiments.md).

## Quick start (synthetic examples)

The examples do **not** need MOSI/MOSEI pickles or a GPU.

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

Individual demos:

```bash
python examples/demo_fusion.py
python examples/demo_gmtm.py
python examples/demo_metrics.py
python examples/demo_train_toy.py
python examples/inspect_shapes.py
```

See [`examples/README.md`](examples/README.md).

## Reproducing the original tables

1. Place aligned pickles under `model/data/MOSEI/` and `model/data/MOSI/`
   (file names are listed in [`docs/datasets.md`](docs/datasets.md)).
2. From `model/`:

```bash
python train_main_bert.py          # fusion sweep, BERT
python train_main_glove.py         # fusion sweep, GloVe
python train_GMTM_bert.py          # gated transformer + ablation
python train_GMTM_glove.py
```

Training uses `AdamW`, `lr=1e-4`, `weight_decay=0.01`, `L1Loss` on the
continuous score, gradient clip `8`, and early stopping on validation L1
(patience 7). Full flags: [`docs/training.md`](docs/training.md).

## Personal notes

- License: MIT (see [`LICENSE`](LICENSE)).
- The class name `GatedMultiTransfomerModel` is misspelled in the original
  source (`Transfomer` vs `Transformer`). Docs keep the code identifier so
  imports match.
- Several training scripts still contain commented-out `train(...)` calls and
  load `checkpoints/*.pt` that are **not** in git (only placeholder readmes).
  You cannot replay the published numbers without the pickles and the
  checkpoint files from the original machine.
- MOSI scripts under `model/mosi_test/` evaluate MOSEI-trained checkpoints on
  MOSI. `get_mosi_dataloader` concatenates MOSI train+valid+test into one
  evaluation pool — that is intentional for the transfer note, not a standard
  MOSI test split.

## Citation-style references

The fusion modules follow published methods; this repo is a personal
re-implementation for comparison, not an official release of those papers.

- Zadeh et al., *Tensor Fusion Network for Multimodal Sentiment Analysis* (EMNLP 2017)
- Liu et al., *Efficient Low-rank Multimodal Fusion with Modality-Specific Factors* (ACL 2018)
- Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences* (ACL 2019)
- Zadeh et al., *Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Spoken Words* (MOSI)
- Zadeh et al., *Multimodal Language Analysis in the Wild: CMU-MOSEI* (ACL 2018)
