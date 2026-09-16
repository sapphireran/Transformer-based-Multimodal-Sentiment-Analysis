# Checkpoints for the main fusion sweep

This directory is a **local** weight dump. `*.pt` files are gitignored.

Expected names written by `train_main_bert.py` / `train_main_glove.py`:

| File | Script | Features |
| --- | --- | --- |
| `ConcatEarly.pt` | `train_main_bert.py` | BERT |
| `ConcatLate.pt` | `train_main_bert.py` | BERT |
| `LowRankTensorFusion.pt` | `train_main_bert.py` | BERT |
| `TensorFusion.pt` | `train_main_bert.py` | BERT |
| `TransformerEarly.pt` | `train_main_bert.py` | BERT |
| `TransformerLate.pt` | `train_main_bert.py` | BERT |
| `glove_<Fusion>.pt` | `train_main_glove.py` | GloVe |

Format: `torch.save(MultiFramework, path)` (full module pickle), not a
`state_dict`. Load with the same `models.py` + `train_and_test.py`.

MOSI transfer scripts look for these files at `../checkpoints/` relative
to `model/mosi_test/`.
