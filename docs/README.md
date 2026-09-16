# Documentation index

These notes describe the **personal** multimodal sentiment experiments in this
repository. They are written against the code that actually lives under
`model/`, not against a cleaned-up library API.

Read in this order if you are new to the repo:

1. [Repository map](repo_map.md) — which file does what
2. [Datasets](datasets.md) — MOSI / MOSEI, feature widths, pickle layout
3. [Architecture](architecture.md) — encoders, `MultiFramework`, GMTM
4. [Fusion methods](fusion_methods.md) — the six sweep recipes plus GMTM
5. [Training](training.md) — scripts, hyperparameters, packed vs padded
6. [Evaluation](evaluation.md) — MAE, Acc7/5/2, F1, correlation
7. [Results](results.md) — numbers copied from `model/results/*.csv`
8. [Ablation](ablation.md) — leave-one-modality-out interpretation
9. [Reproduction](reproduction.md) — how to rerun, what is missing from git
10. [Glossary](glossary.md) — short definitions

Runnable CPU toys that do **not** need the CMU pickles:

- [`../examples/README.md`](../examples/README.md)

## Scope

In scope:

- How visual / audio / text tensors move through each fusion recipe
- How `train()` / `test()` compute losses and metrics
- What the checked-in CSV tables mean

Out of scope:

- Company or product code
- Hosted training services
- Publishing new MOSI / MOSEI leaderboard numbers (the CSVs are personal runs)

## Conventions used in the notes

- Tensor shapes use `B` batch, `T` time, `F` feature.
- “BERT text” means `F_text = 768`. “GloVe text” means `F_text = 300`.
- Visual is always Facet-42 reduced to **35** dims in these pickles.
- Audio is always COVAREP reduced to **74** dims.
- Sentiment labels are treated as **regression** targets (`L1Loss`) unless a
  note explicitly talks about binarization for Acc2 / F1.
