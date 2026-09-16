# Checkpoints — main fusion sweep

`train_main_bert.py` and `train_main_glove.py` write **whole**
`MultiFramework` objects here with `torch.save(model, path)` (not a
`state_dict`). Weight files are gitignored.

| File | Embedding | Fusion |
| --- | --- | --- |
| `ConcatEarly.pt` | BERT | early concat + LSTM |
| `ConcatLate.pt` | BERT | late LSTM concat |
| `LowRankTensorFusion.pt` | BERT | LRTF rank 32 |
| `TensorFusion.pt` | BERT | full TFN |
| `TransformerEarly.pt` | BERT | early transformer |
| `TransformerLate.pt` | BERT | late transformer |
| `glove_ConcatEarly.pt` | GloVe | early concat + LSTM |
| `glove_ConcatLate.pt` | GloVe | late LSTM concat |
| `glove_LowRankTensorFusion.pt` | GloVe | LRTF |
| `glove_TensorFusion.pt` | GloVe | TFN |
| `glove_TransformerEarly.pt` | GloVe | early transformer |
| `glove_TransformerLate.pt` | GloVe | late transformer |

Reload from `model/` so `models.py` is importable:

```python
import torch
model = torch.load("checkpoints/TransformerLate.pt", map_location="cpu")
```

MOSI scripts look one directory up: `../checkpoints/{Fusion}.pt`.
Ablation / GMTM weights live in [`ablation/`](ablation/readme.md).
