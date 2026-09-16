# Transformer-based Multimodal Sentiment Analysis

Personal research code for **multimodal sentiment regression** on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
The repo compares a gated cross-modal transformer (**GMTM**) with standard
early / late / tensor fusion baselines, using either **BERT** or **GloVe**
text features plus Facet visual and COVAREP acoustic streams.

This is a personal project. It is not company code.

## What is in here

| Path | Role |
| --- | --- |
| [`model/models.py`](model/models.py) | Encoders, fusion modules, GMTM |
| [`model/train_and_test.py`](model/train_and_test.py) | `MultiFramework` trainer and affect metrics |
| [`model/data/get_dataloader.py`](model/data/get_dataloader.py) | MOSI / MOSEI pickle loaders and collate |
| [`model/train_main_bert.py`](model/train_main_bert.py) | MOSEI BERT fusion sweep |
| [`model/train_main_glove.py`](model/train_main_glove.py) | MOSEI GloVe fusion sweep |
| [`model/train_GMTM_bert.py`](model/train_GMTM_bert.py) | MOSEI GMTM + BERT ablation |
| [`model/train_GMTM_glove.py`](model/train_GMTM_glove.py) | MOSEI GMTM + GloVe ablation |
| [`model/mosi_test/`](model/mosi_test/) | Transfer / eval of MOSEI checkpoints on MOSI |
| [`model/results/`](model/results/) | Logged tables from the main runs |
| [`docs/`](docs/) | Architecture, data, metrics, reproduction notes |
| [`examples/`](examples/) | CPU-runnable synthetic walkthroughs |

## Feature layout

Every clip is a triple of aligned sequences plus a scalar sentiment score
in roughly `[-3, 3]`.

| Stream | Source | Dim (BERT runs) | Dim (GloVe runs) |
| --- | --- | --- | --- |
| Visual | Facet 4.2 | 35 | 35 |
| Audio | COVAREP | 74 | 74 |
| Text | BERT / GloVe | 768 | 300 |

Sequence length is truncated / padded to **50** frames when `max_pad=True`
(the transformer / GMTM path). Packed-length batches are used by the LSTM
and GRU fusion baselines.

## Fusion methods

- **ConcatEarly** — concatenate features on the last axis, then an LSTM + MLP.
- **ConcatLate** — one LSTM per modality, concatenate last hidden states.
- **TensorFusion** — outer product of `[1; z_m]` across modalities.
- **LowRankTensorFusion** — rank-`r` factorization of the same tensor.
- **TransformerEarly** — 1×1 conv into a shared encoder over concatenated features.
- **TransformerLate** — per-modality `TransformerSeq`, then a late transformer.
- **GMTM** (`GatedMultiTransfomerModel`) — pairwise cross-modal attention,
  learned modality weights, a sigmoid gate, attention pooling, and an MLP head.

See [docs/architecture.md](docs/architecture.md) for tensor shapes and the
forward pass.

## Headline MOSEI (BERT) numbers

Copied from [`model/results/main_results.csv`](model/results/main_results.csv).
Lower MAE is better; higher Acc / Corr / F1 is better.

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

GMTM ablation on the same BERT features is in
[`model/results/ablation_results.csv`](model/results/ablation_results.csv).
Text carries most of the signal; audio-only and visual-only stay near
chance on 7-way accuracy. Full tables live in
[docs/experiments.md](docs/experiments.md).

## Metrics in one sentence

The head predicts a **scalar** sentiment. We report MAE / MSE / Pearson
correlation on that scalar, Acc7 / Acc5 after uniform binning of `[-3, 3]`,
and binary Acc2 / F1 after dropping zero labels. Details:
[docs/evaluation.md](docs/evaluation.md).

## Run the synthetic examples (no dataset required)

```bash
python3 -m pip install -r requirements.txt
python3 examples/fusion_forward.py
python3 examples/gmtm_forward.py
python3 examples/gmtm_toy_train.py
python3 examples/evaluate_metrics.py
python3 -m pytest tests -q
```

These scripts build tiny random batches with the same rank and layout as
the real loaders, so you can inspect shapes and a short training loop
without downloading MOSI / MOSEI. Walkthrough:
[examples/README.md](examples/README.md).

## Train on the real pickles

1. Place `mosei_raw_bert.pkl` / `mosei_raw_glove.pkl` under `model/data/MOSEI/`.
2. Place `mosi_raw_bert.pkl` / `mosi_raw_glove.pkl` under `model/data/MOSI/`.
3. From `model/`, uncomment the `train(...)` call in the script you want
   and run it. Checkpoints are written under `model/checkpoints/`.

Raw CMU SDK files (`.csd`) and the GloVe `840B.300d` vectors are listed in
[`model/data/readme.md`](model/data/readme.md). A longer setup note is in
[docs/data-pipeline.md](docs/data-pipeline.md) and
[docs/reproducing.md](docs/reproducing.md).

## Personal docs

- [Architecture](docs/architecture.md)
- [Data pipeline](docs/data-pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Experiment catalog](docs/experiments.md)
- [Hyperparameters](docs/hyperparameters.md)
- [Reproduction](docs/reproducing.md)
- [File map](docs/file-map.md)
- [Glossary](docs/glossary.md)

## License

MIT. See [LICENSE](LICENSE).
