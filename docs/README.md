# Documentation

These notes describe **this** personal codebase, not a generic multimodal
toolkit. Paths, tensor shapes, and numbers are taken from the scripts and
CSV files that already live under `model/`.

## Suggested reading order

1. [repository-map.md](repository-map.md) — where each file sits and what it
   actually imports.
2. [architecture.md](architecture.md) — the `encoders → fusion → head`
   contract and the gated multi-transformer.
3. [fusion-methods.md](fusion-methods.md) — Concat / TFN / LMF / transformer
   early & late, with shapes.
4. [datasets.md](datasets.md) — MOSI / MOSEI pickle schema, alignment,
   padding, and how ablation zeros unused modalities.
5. [training.md](training.md) — `train()`, packed vs `max_pad`, script entry
   points.
6. [evaluation.md](evaluation.md) — the exact Acc-7 / Acc-5 / Acc-2 / F1
   definitions used in `train_and_test.py`.
7. [hyperparameters.md](hyperparameters.md) — encoder widths and GMTM
   `HParams`.
8. [results.md](results.md) and [ablation-study.md](ablation-study.md) — the
   published tables.
9. [reproducing-experiments.md](reproducing-experiments.md) — a checklist if
   you have the pickle dumps.

Runnable stand-ins that do not need those dumps are under
[`../examples/`](../examples/README.md).

## Conventions used throughout

- Batch-first tensors unless a transformer encoder internally permutes to
  `(T, B, F)`.
- Modality order in lists is always **vision, audio, text** (indices 0, 1, 2
  in `get_dataloader.py`).
- BERT experiments use text width **768**. GloVe experiments use **300**.
- Sentiment is a **scalar regression** in roughly `[-3, 3]`, optimized with
  `L1Loss`. Classification numbers are derived after the fact.

## What this documentation is not

- It is not a replacement for the CMU MultiComp dataset cards.
- It does not claim that the uniform Acc-7 / Acc-5 bins match every paper
  that uses MOSI's original uneven bins. They do not; see evaluation.md.
- It does not vendor MultiBench. Several class names (`TensorFusion`,
  `LowRankTensorFusion`, `MultiFramework`) follow that lineage, but the
  training scripts and GMTM are local to this repo.
