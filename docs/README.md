# Personal notes for this repo

These pages are the long-form lab notes for
`Transformer-based-Multimodal-Sentiment-Analysis`. They describe the code
that is actually in `model/`, not a cleaned-up rewrite.

| Note | Contents |
| --- | --- |
| [Architecture](architecture.md) | Encoders, fusion modules, GMTM data flow, tensor shapes |
| [Datasets](datasets.md) | MOSI / MOSEI files, pickle schema, alignment, ablation masking |
| [Metrics](metrics.md) | MAE / Corr / Acc2 / Acc5 / Acc7 / F1 and how bins are cut |
| [Training](training.md) | `MultiFramework`, losses, early stopping, script flags |
| [Experiments](experiments.md) | Every CSV table plus a conservative reading of the numbers |
| [Reproduction](reproduction.md) | What you need on disk to rerun vs. what the examples cover |
| [Repo map](repo-map.md) | File-by-file index |
| [Quirks](quirks.md) | Known inconsistencies in the original scripts |
| [References](references.md) | Papers the modules follow |

Runnable, dataset-free walkthroughs live in [`../examples/`](../examples/README.md).
