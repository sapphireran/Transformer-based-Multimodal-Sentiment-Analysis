# Transformer-based Multimodal Sentiment Analysis

Personal research repo for **gated cross-modal transformers** on CMU-MOSI and CMU-MOSEI. The code compares a custom **Gated Multi-Transformer (GMTM)** against early/late concatenation, tensor fusion, low-rank tensor fusion, and transformer fusion baselines.

Sentiment is treated as **regression on [-3, 3]** (L1 loss). Classification numbers (Acc-2 / Acc-5 / Acc-7 / F1) are derived after the fact by binning the continuous scores.

| Item | Value |
| --- | --- |
| Modalities | visual (FACET 4.2, 35-d), acoustic (COVAREP, 74-d), text (BERT 768-d or GloVe 300-d) |
| Datasets | [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) (main), [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) (transfer / extra test) |
| Fusion study | ConcatEarly, ConcatLate, TensorFusion, LowRankTensorFusion, TransformerEarly, TransformerLate, GMTM |
| Text backends | BERT (`*_bert.py`) and GloVe (`*_glove.py`) |
| License | MIT |

This tree is **personal research code only**. It is not a product, not a company codebase, and not a drop-in replacement for MultiBench / MulT.

## What is in the repo

```text
model/
  models.py                 Fusion blocks + GMTM
  train_and_test.py         Supervised loop, metrics, MultiFramework wrapper
  train_main_bert.py        MOSEI BERT fusion baselines
  train_main_glove.py       MOSEI GloVe fusion baselines
  train_GMTM_bert.py        MOSEI BERT GMTM / modality ablation
  train_GMTM_glove.py       MOSEI GloVe GMTM / modality ablation
  data/get_dataloader.py    MOSI / MOSEI pickle loaders
  data/MOSEI/               BERT / GloVe pickle builders + CMU-SDK notes
  data/MOSI/                MOSI pickle builders
  mosi_test/                Evaluate MOSEI checkpoints on MOSI
  results/                  Published CSV tables + plot notebook
  checkpoints/              Where .pt files are written (not checked in)
docs/                       Architecture, data, training, metrics, results
examples/                   CPU-only synthetic forward passes and metric demos
```

Start with the docs if you are reading rather than training:

- [Architecture](docs/architecture.md) — encoders, fusion family, GMTM internals
- [Datasets](docs/datasets.md) — MOSI / MOSEI features, pickle layout, alignment
- [Training](docs/training.md) — scripts, hyperparameters, packed vs padded batches
- [Evaluation](docs/evaluation.md) — MAE / Corr / Acc-2 / Acc-5 / Acc-7 / F1
- [Results](docs/results.md) — tables transcribed from `model/results/*.csv`
- [Reproducing experiments](docs/reproducing.md) — data prep and command list
- [Repository map](docs/repository_map.md) — every script and what it does
- [Examples](examples/README.md) — runnable CPU demos that do not need MOSI/MOSEI

## Headline MOSEI (BERT) numbers

From `model/results/main_results.csv` plus the full-modality GMTM row in `model/results/ablation_results.csv`:

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (text+audio+visual) | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

On this BERT / MOSEI sweep, **GMTM is best on MAE, Acc-7, Acc-2, correlation, and F1**. Late transformer fusion is the strongest baseline. Text carries most of the signal; audio-only and visual-only ablations collapse (see [results](docs/results.md)).

## Quick start (examples, no dataset)

The examples construct MOSI/MOSEI-shaped tensors and run the real modules from `model/models.py`. They stay on CPU and do not download GloVe, BERT, or CMU SDK files.

```bash
python -m pip install -r requirements.txt
python examples/run_all_examples.py
python -m pytest tests/test_examples.py -q
```

Individual demos:

```bash
python examples/fusion_forward.py
python examples/gmtm_forward.py
python examples/metrics_demo.py
python examples/summarize_results.py
```

## Train a baseline (needs local pickles)

Place aligned pickles under `model/data/MOSEI/` (see [datasets](docs/datasets.md)), then from `model/`:

```bash
# MOSEI BERT fusion sweep (writes checkpoints/<Fusion>.pt and main_results.csv)
python train_main_bert.py

# MOSEI BERT GMTM (uncomment the train(...) call in train_GMTM_bert.py first)
python train_GMTM_bert.py
```

Default training recipe used by the BERT main sweep:

- optimizer: AdamW, `lr=1e-4`, `weight_decay=0.01`
- loss: `L1Loss` on the raw score
- batch size: 32
- sequence cap: 50 steps
- early fusion transformer uses `max_pad=True`; other fusions use packed lengths
- checkpoints: `model/checkpoints/` and `model/checkpoints/ablation/`

`train_main_bert.py` currently passes `total_epochs=1`. Raise that (and restore the commented `train(...)` calls in the GloVe / GMTM scripts) before a real run. Details are in [training](docs/training.md).

## Feature dimensions

| Modality | Source | Dim |
| --- | --- | ---: |
| Visual | FACET 4.2 (aligned to words) | 35 |
| Acoustic | COVAREP | 74 |
| Text (BERT) | last-layer word pieces / aligned BERT | 768 |
| Text (GloVe) | `glove.840B.300d` | 300 |

Early-fusion BERT input width is `35 + 74 + 768 = 877`. Early-fusion GloVe width is `35 + 74 + 300 = 409`.

## Metrics in one paragraph

`single_test` in `model/train_and_test.py` reports:

1. **MAE / MSE / Pearson r** on the continuous prediction vs. gold score.
2. **Acc-7 / Acc-5** after slicing `[-3, 3]` into 7 or 5 equal-width bins (`split_uniform_7` / `split_uniform_5`).
3. **Acc-2 / binary F1** after mapping `score > 0` vs. `score < 0`, optionally dropping zeros (`eval_affect`).

MOSI transfer loaders (`get_mosi_dataloader`) concatenate train+valid+test into one evaluation pool so a MOSEI checkpoint can be scored on every MOSI clip.

## Status / caveats

- Checkpoints and raw `.csd` / `.pkl` files are **not** in git. Only CSV summaries are.
- Several GloVe / GMTM scripts have `train(...)` commented and only load an existing `.pt`.
- `train_main_glove.py` and some MOSI testers still pass kwargs (`dataset=`, `no_robust=`) that `test()` no longer accepts — strip those before running.
- `model/data/MOSEI/get_mosei.py` is a MOSI CMU-SDK alignment script with hard-coded Windows paths; treat it as a personal notebook, not a portable CLI.
- `models.py` defines both a `Linear` class and a later `Linear(...)` factory; the factory wins at import time.

## License

MIT. Copyright (c) 2024 pang990801. See [LICENSE](LICENSE).
