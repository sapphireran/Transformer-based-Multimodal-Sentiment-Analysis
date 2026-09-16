# Documentation

Personal notes for this multimodal sentiment project. Nothing here is
company documentation. The experiment scripts under `model/` are the
source of truth; these pages describe what those scripts actually do.

| Page | Contents |
| --- | --- |
| [architecture.md](architecture.md) | Encoders, fusion blocks, GMTM forward pass |
| [datasets.md](datasets.md) | CMU-MOSI / CMU-MOSEI files, pickle schema, loaders |
| [training.md](training.md) | `MultiFramework`, hyperparameters, script map |
| [evaluation.md](evaluation.md) | MAE, Acc-5/7, binary F1, bin edges |
| [results.md](results.md) | Numbers copied from the CSV tables |
| [reproducibility.md](reproducibility.md) | Hardware, seeds, missing artifacts |
| [glossary.md](glossary.md) | Short lexicon |

If you just want to poke the models without downloading datasets, skip
to [`../examples/README.md`](../examples/README.md).
