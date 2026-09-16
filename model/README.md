# `model/` — training code

This directory is the original working tree. Launch scripts from
**here** so the relative pickle and checkpoint paths resolve.

| Script | Typical command |
| --- | --- |
| BERT fusion sweep | `python train_main_bert.py` |
| GloVe fusion sweep | `python train_main_glove.py` |
| BERT GMTM | `python train_GMTM_bert.py` |
| GloVe GMTM | `python train_GMTM_glove.py` |
| MOSI transfer | `python mosi_test/train_mosi_bert.py` |

Several scripts have `train(...)` commented out and only reload a
`.pt` file. That is the re-eval path. Uncomment the train block for
a fresh fit. Full operator notes: [docs/training.md](../docs/training.md).

Architecture of `models.py`: [docs/architecture.md](../docs/architecture.md).
What the CSVs mean: [docs/results.md](../docs/results.md).

If you only want to touch the modules without a dataset, use the
CPU demos in [`../examples/`](../examples/README.md) instead of
editing these scripts first.
