# Model storage — ablation study

GMTM checkpoints from `train_GMTM_bert.py` and `train_GMTM_glove.py`.
Names encode the **kept** modalities; the other streams were zeros.

| File pattern | Text features |
| --- | --- |
| `model_{mod+mod}.pt` | BERT |
| `model_glove_{mod+mod}.pt` | GloVe |

Example: `model_text+audio+visual.pt`.

Tables for these runs: [docs/experiments.md](../../../docs/experiments.md).
