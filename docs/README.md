# Documentation index

Personal notes for this multimodal sentiment repo. They describe the code that is already here; they are not a company design doc.

| Doc | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | `MultiFramework`, encoders, GMTM forward path, tensor layouts |
| [fusion-methods.md](fusion-methods.md) | Concat, tensor, low-rank, transformer early/late |
| [datasets.md](datasets.md) | MOSI / MOSEI features, pickle layout, dataloader flags |
| [training.md](training.md) | Scripts, hyperparameters, checkpoint paths |
| [evaluation.md](evaluation.md) | MAE, correlation, Acc-2/5/7, F1, uniform bins |
| [ablation.md](ablation.md) | Modality drop protocol and how zeros are injected |
| [results.md](results.md) | Recorded CSV tables and a short reading of the numbers |
| [reproducing.md](reproducing.md) | How to rebuild pickles and rerun the original scripts |
| [scripts-map.md](scripts-map.md) | File-by-file map of training / test entry points |

Runnable walkthroughs that do not need the CMU SDK live in [`../examples/`](../examples/README.md).
