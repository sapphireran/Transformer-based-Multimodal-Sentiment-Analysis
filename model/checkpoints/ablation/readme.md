# Ablation-study checkpoints

GMTM files written by `train_GMTM_bert.py` and `train_GMTM_glove.py`.
Each file is a full 3-stream `GatedMultiTransfomerModel` wrapped in
`MultiFramework`. Unused modalities were **zeroed** at train time, not
removed (see `docs/ablation-study.md`).

| Pattern | Example |
| --- | --- |
| `model_{modalities}.pt` | `model_text+audio+visual.pt`, `model_text.pt` |
| `model_glove_{modalities}.pt` | `model_glove_text+visual.pt` |

`modalities` is `'+'.join(...)` in the order listed in the script
(`text`, `audio`, `visual`, and the pairwise / triple combinations).

`.pt` files are gitignored. Scores are in
`model/results/ablation_results.csv` and
`model/results/ablation_glove_results.csv`.
