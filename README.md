# Transformer-based Multimodal Sentiment Analysis

Personal research code for **gated multi-transformer fusion** (GMTM) and a set of
standard multimodal baselines on [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
and [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).

The project compares how text, audio, and visual streams are combined for
continuous sentiment regression on the usual `[-3, 3]` Likert scale. Text can be
encoded with **BERT** (`768-d`) or **GloVe 840B** (`300-d`). Visual features are
FACET 4.2 (`35-d`). Acoustic features are COVAREP (`74-d`). Sequences are clipped
or padded to **50** aligned time steps.

This repository is a personal study project. It is not affiliated with any
company. The original training scripts live under `model/`. The `docs/` and
`examples/` trees added here explain the architecture, data contract, metrics,
and recorded tables, and they ship CPU-only walkthroughs that do **not** need
the full MOSI/MOSEI pickles or a GPU.

## Why GMTM

Baseline fusion in this repo is either:

- **early** — concatenate raw (or lightly projected) streams, then encode
- **late** — encode each stream, then concatenate or take an outer product
- **tensor** — TFN / low-rank multimodal fusion over per-modality vectors

GMTM instead:

1. Projects each modality into a shared `embed_dim`
2. Runs **pairwise cross-modal transformers** (`i` attends to `j` for every pair)
3. Mixes the pair outputs with **learned modality weights**
4. Applies a **sigmoid gate** per modality
5. Concatenates the gated streams and **attention-pools over time**
6. Predicts a scalar sentiment score

On the recorded MOSEI + BERT ablation, the full text+audio+visual GMTM is the
strongest row (MAE `0.5640`, Acc-2 `0.8429`, F1 `0.8777`). Text alone is already
close; audio and visual without text collapse. That is the main empirical story
of the project. Details and caveats are in [`docs/results.md`](docs/results.md).

## Repository map

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, GMTM |
| [`model/train_and_test.py`](model/train_and_test.py) | `MultiFramework`, train loop, MOSI/MOSEI metrics |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | Pickle loaders, padding, modality dropout for ablation |
| [`model/train_main_bert.py`](model/train_main_bert.py) / [`train_main_glove.py`](model/train_main_glove.py) | Baseline sweep on MOSEI |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) / [`train_GMTM_glove.py`](model/train_GMTM_glove.py) | GMTM + modality ablation on MOSEI |
| [`model/mosi_test/`](model/mosi_test/) | Evaluate MOSEI-trained checkpoints on MOSI pickles |
| [`model/results/`](model/results/) | Canonical CSV tables used in the docs |
| [`docs/`](docs/) | Architecture, data, metrics, experiments, results |
| [`examples/`](examples/) | CPU walkthroughs with synthetic tensors |

A longer file-by-file guide is in [`docs/repo-map.md`](docs/repo-map.md).

## Documentation

- [Architecture](docs/architecture.md) — modules, tensor shapes, GMTM diagram
- [Data pipeline](docs/data-pipeline.md) — pickle layout, alignment, MOSI vs MOSEI
- [Metrics](docs/metrics.md) — MAE, Corr, Acc-7 / Acc-5 / Acc-2, F1
- [Experiments](docs/experiments.md) — how each training/eval script is wired
- [Results](docs/results.md) — copied tables plus a plain-language reading
- [Getting started](docs/getting-started.md) — environment, paths, GPU notes

## Quick start (examples, no dataset)

```bash
python3 -m pip install -r requirements.txt
python3 examples/read_result_tables.py
python3 examples/gmtm_forward.py
python3 examples/fusion_shapes.py
python3 examples/metrics_demo.py
python3 examples/multiframework_demo.py
python3 examples/dataset_collate_demo.py
bash examples/run_all.sh
python3 -m pytest -q
```

The walkthroughs build tiny random batches with the same feature widths as
MOSEI (`35 / 74 / 768` or `300`) so you can see shapes and metric math without
downloading CMU SDK files.

## Train / evaluate the original scripts

From `model/` after the BERT or GloVe pickles are in place
(see [data pipeline](docs/data-pipeline.md)):

```bash
cd model
# MOSEI baselines, BERT text
python train_main_bert.py
# GMTM + modality ablation, BERT text
python train_GMTM_bert.py
```

Scripts assume a CUDA device (`*.cuda()`). Checkpoints are written under
`model/checkpoints/`. Several eval scripts currently **load** a `.pt` file
instead of training; uncomment the `train(...)` call when you want to fit a
new run. MOSI numbers come from `model/mosi_test/` and, as documented, the MOSI
helper concatenates train+valid+test — treat those rows as a transfer check,
not a standard held-out MOSI split.

## License

MIT. See [`LICENSE`](LICENSE).
