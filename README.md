# Transformer-based Multimodal Sentiment Analysis

Personal study repo for clip-level sentiment on [CMU-MOSI](https://arxiv.org/abs/1706.01125) and
[CMU-MOSEI](https://arxiv.org/abs/1802.05799). Three streams — Facet 4.2
(visual, 35-d), COVAREP (audio, 74-d), and BERT-768 or GloVe-300 (language) —
are fused with concatenation, tensor fusion, transformers, and a **gated
cross-modal transformer** (GMTM).

The original upload was a training dump (models, CSVs, notebooks) with a
one-line README. This tree adds documentation, a CPU examples lab that
does **not** need the CMU pickles, and tests for the metric / fusion
helpers.

## Quick start (no dataset)

```bash
pip install -r requirements.txt
python scripts/run_examples.py
python -m pytest
```

That path stays on CPU. It generates synthetic `[B, T, F]` clips, runs
every fusion module once, overfits GMTM for a few AdamW steps, and
prints the same MAE / Acc-7 / Acc-2 / F1 mapping the CSVs use.

Numbered walkthroughs live in [`examples/`](examples/README.md):

1. Synthetic batch shapes for the toy / BERT / GloVe packs
2. Forward-pass output shapes per fusion family
3. GMTM toy overfit (MAE must drop)
4. Metric walkthrough (uniform bins, binary F1)
5. Packed collate vs max-pad collate
6. Ablation by zeroing dropped modalities

## What GMTM is

`GatedMultiTransfomerModel` in `model/models.py`:

1. project each modality to a shared `embed_dim`;
2. run a transformer for every pair `(query i, key/value j)`;
3. mix those pairwise streams with a learned softmax over modalities;
4. apply a sigmoid feature gate;
5. attention-pool over time;
6. regress a single score in `[-3, 3]` with L1 loss.

Details, diagrams, and the encoder–fusion–head wrapper:
[docs/architecture.md](docs/architecture.md).

## Headline numbers (MOSEI BERT)

From [`model/results/main_results.csv`](model/results/main_results.csv)
and the GMTM ablation table. Lower MAE is better.

| fusion | MAE | Acc-7 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- |
| ConcatLate | 0.6148 | 0.4508 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TransformerLate | 0.5846 | 0.4675 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (trimodal) | **0.5640** | **0.4827** | **0.8429** | **0.7255** | **0.8777** |

Language does most of the work (text-only GMTM MAE 0.5687). Audio or
visual alone sit around 0.82 MAE. Full tables, GloVe, and MOSI transfer:
[docs/experiments.md](docs/experiments.md).

## Repository map

```text
docs/                  architecture, datasets, fusion, eval, quirks
examples/              CPU lab (msa_lab + six scripts)
model/models.py        fusion modules + GMTM
model/train_and_test.py  MultiFramework train / test loop
model/metrics.py       headless MAE / Acc-7 / F1
model/data/            dataloaders + SDK notebooks
model/results/         MOSEI CSVs
model/mosi_test/       MOSI transfer scripts + CSVs
tests/                 pytest for lab helpers
```

## Full MOSI / MOSEI training

Needs the CMU pickles (not in git), a GPU, and a few script fixes
documented in [docs/known-quirks.md](docs/known-quirks.md). Recipe:
[docs/reproduction.md](docs/reproduction.md). Feature list:
[docs/datasets.md](docs/datasets.md).

## License

MIT. See [LICENSE](LICENSE).
