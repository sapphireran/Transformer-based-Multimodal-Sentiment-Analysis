# Checkpoints — main experiments

`train_main_bert.py` and `train_main_glove.py` write whole `MultiFramework` pickles here (`torch.save(model, path)`, not a `state_dict`). The files are gitignored.

| File | Script | Text |
| --- | --- | --- |
| `ConcatEarly.pt` | `train_main_bert.py` | BERT |
| `ConcatLate.pt` | `train_main_bert.py` | BERT |
| `LowRankTensorFusion.pt` | `train_main_bert.py` | BERT |
| `TensorFusion.pt` | `train_main_bert.py` | BERT |
| `TransformerEarly.pt` | `train_main_bert.py` | BERT |
| `TransformerLate.pt` | `train_main_bert.py` | BERT |
| `glove_ConcatEarly.pt` | `train_main_glove.py` | GloVe |
| `glove_ConcatLate.pt` | `train_main_glove.py` | GloVe |
| `glove_LowRankTensorFusion.pt` | `train_main_glove.py` | GloVe |
| `glove_TensorFusion.pt` | `train_main_glove.py` | GloVe |
| `glove_TransformerEarly.pt` | `train_main_glove.py` | GloVe |
| `glove_TransformerLate.pt` | `train_main_glove.py` | GloVe |

MOSI testers load the BERT names from `model/mosi_test/` via `../checkpoints/<Fusion>.pt`. Reload with `torch.load(path)` and keep the class layout in `models.py` / `train_and_test.py` stable.

Ablation / GMTM files live in [ablation/](ablation/readme.md).
