# Transformer-based Multimodal Sentiment Analysis

Personal research code for comparing **fusion families** on
[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) and
[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).
The input is always the same three streams — FACET vision, COVAREP audio, and
either BERT or GloVe text — and the target is a continuous sentiment in
`[-3, 3]`.

The model this repo adds is a **gated multi-transformer** (GMTM): pairwise
cross-modal Transformer encoders, a learned modality mix, a per-stream gate,
and attention pooling over time. Baselines include early/late concat, Tensor
Fusion, Low-rank Multimodal Fusion, and early/late Transformers.

On the last MOSEI BERT run, GMTM is the best row in
[`model/main_results.csv`](model/main_results.csv) / the last line of
[`model/ablation_results.csv`](model/ablation_results.csv):

| Method | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- |
| TransformerLate (best baseline) | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM text+audio+visual | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

Acc-7 / Acc-5 use **uniform bins** of `[-3, 3]`, not the integer 7-class
mapping from some older MOSI papers. Details are in
[docs/metrics.md](docs/metrics.md). Full tables: [docs/results.md](docs/results.md).

## Repository layout

```
docs/                  architecture, datasets, training, metrics, results
examples/              CPU scripts that do not need the CMU pickles
tests/                 pytest for metrics + toy forward/train
model/
  models.py            fusion modules + GatedMultiTransfomerModel
  metrics.py           MAE / Acc-7 / Acc-2 / F1 helpers
  train_and_test.py    MultiFramework + train / test loop
  train_main_*.py      six-fusion MOSEI sweeps (BERT / GloVe)
  train_GMTM_*.py      gated multi-transformer + modality ablation
  data/                pickle contract + SDK rebuild notes
  mosi_test/           MOSEI → MOSI transfer scripts
  results/             CSV snapshot used by plot.ipynb
```

## Quick start (no dataset)

```bash
pip install -r requirements-dev.txt
pytest tests/test_metrics.py tests/test_examples.py -q
python examples/run_all.py
```

`examples/` rebuilds `[B, T, 35/74/768|300]` batches in memory, runs every
fusion module, fits a tiny GMTM with AdamW + L1, reprints the committed
CSVs, and writes BERT vs GloVe charts. See
[examples/README.md](examples/README.md). Longer notes:
[docs/README.md](docs/README.md).

## Train on MOSEI (needs pickles + GPU)

```bash
pip install -r requirements.txt
# place mosei_raw_bert.pkl / mosei_raw_glove.pkl under model/data/MOSEI/
cd model
python train_main_bert.py
python train_GMTM_bert.py
```

Feature files, alignment, and the pickle schema:
[model/data/readme.md](model/data/readme.md) and
[docs/datasets.md](docs/datasets.md). Script map and hidden sizes:
[docs/training.md](docs/training.md). Environment caveats:
[docs/reproduction.md](docs/reproduction.md).

## Architecture sketch

```
vision [B,T,35] ─┐
audio  [B,T,74] ─┼─→ per-modality Linear+LN ─→ n×n cross-Transformers
text   [B,T,D]  ─┘         ↓ softmax(modal_weights) + sigmoid gate
                           ↓ concat + attention pool
                           ↓ LayerNorm MLP → scalar in [-3, 3]
```

`D` is 768 (BERT) or 300 (GloVe). Longer write-up:
[docs/architecture.md](docs/architecture.md).

## License

MIT. Copyright (c) 2024 pang990801.
