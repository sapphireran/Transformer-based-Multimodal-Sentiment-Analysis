# Checkpoints — main fusion sweep

`torch.save` dumps of a full `MultiFramework` (not a `state_dict`).
Weights are **not** committed.

| Filename | Text | Method |
| --- | --- | --- |
| `ConcatEarly.pt` | BERT | early concat + LSTM |
| `ConcatLate.pt` | BERT | late LSTM concat |
| `LowRankTensorFusion.pt` | BERT | LMF |
| `TensorFusion.pt` | BERT | TFN |
| `TransformerEarly.pt` | BERT | early transformer |
| `TransformerLate.pt` | BERT | late transformer |
| `glove_ConcatEarly.pt` | GloVe | same six methods |
| `glove_ConcatLate.pt` | GloVe | |
| `glove_LowRankTensorFusion.pt` | GloVe | |
| `glove_TensorFusion.pt` | GloVe | |
| `glove_TransformerEarly.pt` | GloVe | |
| `glove_TransformerLate.pt` | GloVe | |

Produced by `train_main_bert.py` / `train_main_glove.py` when `train()` is
enabled. Load from `model/` so `models.py` and `train_and_test.py` resolve:

```python
import torch
model = torch.load("checkpoints/TransformerLate.pt", map_location="cpu")
```

Ablation / GMTM weights live in [`ablation/`](ablation/readme.md).

Architecture notes: [`../../docs/fusion_methods.md`](../../docs/fusion_methods.md).
