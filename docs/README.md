# Personal documentation index

This folder is the long-form companion to the repository README. It describes the
personal CMU-MOSI / CMU-MOSEI experiments in this repo: what each script does,
how tensors move through the models, how metrics are computed, and how to read
the checked-in CSV tables.

Nothing here is company or product documentation. It is a personal research
lab notebook written against the code that already lives under `model/`.

| Document | What it covers |
| --- | --- |
| [setup.md](setup.md) | Python environment, expected data files, run-from-`model/` convention |
| [architecture.md](architecture.md) | GMTM, fusion modules, encoders, tensor shapes |
| [datasets.md](datasets.md) | MOSI / MOSEI pickle schema, feature dimensions, alignment, ablation masking |
| [metrics.md](metrics.md) | MAE, correlation, Acc7 / Acc5 / Acc2, F1, uniform binning |
| [experiments.md](experiments.md) | How each training and transfer script is wired |
| [results.md](results.md) | Walkthrough of every checked-in CSV |
| [code_map.md](code_map.md) | File-by-file map of `model/` |
| [personal_lab_notes.md](personal_lab_notes.md) | What the tables actually suggest, including caveats |

Runnable companions live in [`examples/`](../examples/README.md). Those scripts
do not need the full MOSI / MOSEI pickles; they use synthetic tensors with the
same rank and feature widths as the real loaders.
