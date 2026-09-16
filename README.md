# Transformer-based Multimodal Sentiment Analysis

Personal research repository for **gated multi-transformer fusion** on aligned
video, audio, and text. The project compares a custom
`GatedMultiTransfomerModel` (GMTM) against early/late concatenation, tensor
fusion, low-rank tensor fusion, and transformer fusion on **CMU-MOSI** and
**CMU-MOSEI**.

This is a personal study repo, not a packaged library. Scripts assume you
already have the official MOSI / MOSEI feature files converted to the pickle
layout described in [`docs/datasets.md`](docs/datasets.md).

## What this repo studies

Spoken-language sentiment is not text-only. A shrug, a pause, or a sarcastic
tone can invert the polarity of the words. MOSI and MOSEI give word-aligned
**visual** (FACET), **acoustic** (COVAREP), and **language** (GloVe or BERT)
sequences plus a continuous sentiment score in roughly `[-3, 3]`.

The modeling question here is: *how should those three streams meet?*

| Family | Idea | Code |
| --- | --- | --- |
| Early concat | Stack features on the channel axis, then one sequence model | `ConcatEarly`, `EarlyFusionTransformer` |
| Late concat | Encode each modality, then concatenate vectors | `ConcatLate`, `LateFusionTransformer` |
| Tensor fusion | Outer-product interactions (full or low-rank) | `TensorFusion`, `LowRankTensorFusion` |
| Gated multi-transformer | Pairwise cross-modal attention, learned modality weights, per-modality gates, attention pooling | `GatedMultiTransfomerModel` |

GMTM is the model this repo is organized around. The other fusions are
baselines trained with the same `MultiFramework` trainer in
`model/train_and_test.py`.

## Repository layout

```
model/
  models.py                 # encoders, fusion modules, GMTM
  train_and_test.py         # MultiFramework, train/eval, MOSI-style metrics
  train_main_bert.py        # MOSEI BERT fusion baselines
  train_main_glove.py       # MOSEI GloVe fusion baselines
  train_GMTM_bert.py        # MOSEI BERT GMTM / modality ablation
  train_GMTM_glove.py       # MOSEI GloVe GMTM / modality ablation
  mosi_test/                # MOSI transfer / eval scripts + CSVs
  results/                  # recorded tables used in the write-up
  data/get_dataloader.py    # pickle -> Affectdataset -> DataLoader
  data/MOSEI/, data/MOSI/   # raw-feature download / align notebooks
docs/                       # architecture, data, metrics, experiments
examples/                   # CPU-friendly synthetic demos (no MOSI/MOSEI files)
```

Recorded scores live in `model/results/` and `model/mosi_test/`. A walkthrough
of those tables is in [`docs/results.md`](docs/results.md).

## Feature shapes this code expects

Aligned clips are truncated / padded to **50** time steps.

| Stream | Source | Dim (BERT setup) | Dim (GloVe setup) |
| --- | --- | --- | --- |
| Visual | FACET 4.2 | 35 | 35 |
| Audio | COVAREP | 74 | 74 |
| Text | BERT / GloVe-840B | 768 | 300 |
| Label | opinion / sentiment | scalar in `[-3, 3]` | same |

`get_ablation_dataloader` keeps all three tensors in the batch but **zeros**
the unused streams so GMTM can stay a 3-input model during modality ablations.

## Quick start (synthetic, no dataset files)

The examples do **not** need MOSI/MOSEI pickles or a GPU. From the repo root:

```bash
python3 -m pip install -r requirements.txt
python3 examples/run_all.py
```

That runner:

1. Builds a tiny aligned multimodal batch (`examples/synthetic_batch.py`).
2. Runs GMTM and the fusion modules on CPU (`examples/gmtm_forward.py`,
   `examples/fusion_shapes.py`).
3. Recomputes MAE / Corr / Acc-7 / Acc-5 / Acc-2 / F1 the same way
   `train_and_test.py` does (`examples/metrics_walkthrough.py`).
4. Prints the logged CSV tables (`examples/summarize_logged_results.py`).
5. Overfits a toy GMTM for a few steps (`examples/tiny_overfit.py`).

## Training on real MOSI / MOSEI

1. Place aligned pickles as documented in [`docs/datasets.md`](docs/datasets.md):
   - `model/data/MOSEI/mosei_raw_bert.pkl`
   - `model/data/MOSEI/mosei_raw_glove.pkl`
   - `model/data/MOSI/mosi_raw_bert.pkl`
   - `model/data/MOSI/mosi_raw_glove.pkl`
2. Install the extra training extras: `pip install -r requirements-train.txt`.
3. From `model/`, run one of the `train_*.py` entry points. Default loss is
   `L1Loss`, optimizer `AdamW`, `lr=1e-4`, `weight_decay=0.01`.
4. Checkpoints are written under `model/checkpoints/` (main) or
   `model/checkpoints/ablation/` (GMTM / modality drops).

Script-by-script flags and expected outputs:
[`docs/experiments.md`](docs/experiments.md).

Evaluation protocol (why Acc-7 is *uniform bins*, not the usual MOSI
integer buckets): [`docs/metrics.md`](docs/metrics.md).

## Headline numbers (already logged)

MOSEI + BERT, full three modalities, GMTM vs strongest transformer baseline:

| Model | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (T+A+V) | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

On that split, **text already carries most of the signal**; audio and vision
help a little, and they are weak alone. See [`docs/research-notes.md`](docs/research-notes.md)
for the GloVe and MOSI caveats (GMTM is not uniformly best).

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — GMTM block diagram and
  baseline fusion math as implemented.
- [`docs/datasets.md`](docs/datasets.md) — MOSI/MOSEI files, pickle keys,
  alignment, z-norm, padding.
- [`docs/metrics.md`](docs/metrics.md) — regression + binned classification.
- [`docs/experiments.md`](docs/experiments.md) — how each training script
  wires encoders / fusion / head.
- [`docs/results.md`](docs/results.md) — every logged CSV, transcribed.
- [`docs/reproduction.md`](docs/reproduction.md) — environment, paths, gotchas.
- [`docs/research-notes.md`](docs/research-notes.md) — personal notes on
  what the tables actually imply.
- [`examples/README.md`](examples/README.md) — runnable CPU demos.

## License

MIT. Copyright (c) 2024 pang990801. See [`LICENSE`](LICENSE).
