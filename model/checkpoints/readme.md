# Model storage location for main experiments

This directory is a placeholder. Trained `MultiFramework` objects are written
with `torch.save(model, path)` from the sweep scripts. Files are not committed.

| File | Script | Fusion |
| --- | --- | --- |
| `ConcatEarly.pt` | `train_main_bert.py` | early concat + LSTM head |
| `ConcatLate.pt` | `train_main_bert.py` | per-stream LSTM + late concat |
| `LowRankTensorFusion.pt` | `train_main_bert.py` | LMF |
| `TensorFusion.pt` | `train_main_bert.py` | TFN |
| `TransformerEarly.pt` | `train_main_bert.py` | `EarlyFusionTransformer` |
| `TransformerLate.pt` | `train_main_bert.py` | per-stream `TransformerSeq` + late transformer |
| `glove_ConcatEarly.pt` … `glove_TransformerLate.pt` | `train_main_glove.py` | same names, GloVe text tower |

MOSI eval scripts load these with a `../checkpoints/` prefix. Some MOSI files
look for `Concat.pt` instead of `ConcatLate.pt` — symlink or rename if you
revive that row.

Ablation / GMTM weights live in [`ablation/`](ablation/readme.md).
