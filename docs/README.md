# Personal notes for this repository

These notes describe the **personal** MOSI / MOSEI transformer-fusion
experiments in this repo. They track how the code is actually written, not an
idealized paper rewrite.

| Note | Contents |
| --- | --- |
| [architecture.md](architecture.md) | `MultiFramework`, encoders, GMTM |
| [fusion-methods.md](fusion-methods.md) | Concat / tensor / transformer fusion |
| [datasets.md](datasets.md) | Feature files, alignment, loaders |
| [training-and-evaluation.md](training-and-evaluation.md) | Loop, losses, Acc-7/5/2 |
| [experiments-and-results.md](experiments-and-results.md) | Commentary on the CSVs |
| [reproduction.md](reproduction.md) | Rerun, ablate, or add a fusion |
| [code-map.md](code-map.md) | Every script and what it calls |
| [glossary.md](glossary.md) | Short MSA glossary |

Runnable, dataset-free companions live in [`../examples/`](../examples).
