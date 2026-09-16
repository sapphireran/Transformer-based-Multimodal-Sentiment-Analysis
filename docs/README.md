# Personal notes for this repo

These pages describe the code that is already in `model/`, the pickle
contract the training scripts expect, and the CSVs that were logged
from personal MOSI / MOSEI runs. They are not a third-party paper.

| Page | Contents |
| --- | --- |
| [architecture.md](architecture.md) | GMTM, encoders, fusion modules, leftover bugs |
| [datasets.md](datasets.md) | MOSI / MOSEI files, pickle keys, alignment, ablation zeros |
| [metrics.md](metrics.md) | MAE, Pearson r, uniform Acc-7/5, exclude-zero Acc-2 / F1 |
| [experiments.md](experiments.md) | What each `train_*.py` wires up |
| [results.md](results.md) | Transcribed CSVs |
| [research-notes.md](research-notes.md) | What those tables imply (and what they do not) |
| [reproduction.md](reproduction.md) | Install tiers, paths, checkpoint loading |
| [glossary.md](glossary.md) | GMTM / FACET / COVAREP / packed / pooled MOSI |

Runnable CPU demos (no dataset download): [`../examples/README.md`](../examples/README.md).

Start at the root [`../README.md`](../README.md) if you have not read
anything else.
