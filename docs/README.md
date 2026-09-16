# Docs index

Personal write-ups for this multimodal sentiment project. They describe the
code that is already in `model/`, including a few sharp edges (commented
`train()` calls, extra `test()` kwargs, MOSI merged splits). Start at the
root [README](../README.md) if you want the short version and the headline
MOSEI BERT GMTM numbers.

| Note | Contents |
| --- | --- |
| [architecture.md](architecture.md) | GMTM internals (projection, n² cross-modal encoders, gate, attention pool) and the six fusion baselines with tensor ranks |
| [datasets.md](datasets.md) | MOSI / MOSEI pickles, FACET / COVAREP / BERT / GloVe widths, padding, aligned slicing, ablation zeros |
| [metrics.md](metrics.md) | MAE, Pearson, drop-zero Acc-2 / F1, uniform Acc-5 / Acc-7, why `plt.show` is avoided in examples |
| [experiments.md](experiments.md) | Every committed CSV (MOSEI + MOSI, BERT + GloVe, baselines + GMTM ablations) and a short reading of each table |
| [training.md](training.md) | `MultiFramework` loop, AdamW / L1 / clip / early-stop, how each script wires encoders |
| [hyperparameters.md](hyperparameters.md) | GMTM `HParams` (64 / 4 / 4) vs the tiny example bundle, encoder widths per fusion |
| [reproduction.md](reproduction.md) | Directory layout, how to re-run, known script mismatches |
| [module_reference.md](module_reference.md) | Class-by-class index for `models.py`, `train_and_test.py`, loaders |
| [repository_map.md](repository_map.md) | Tree after the docs / examples expansion |
| [glossary.md](glossary.md) | Short definitions as this repo uses them |

Runnable companions (synthetic tensors, no CMU download) live in
[`../examples/`](../examples/README.md). The pytest suite under `../tests/`
pins the headline CSV winners so a silent table edit fails CI-style checks.
