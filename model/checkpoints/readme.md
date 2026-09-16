# Checkpoints for the main fusion sweeps

`train_main_bert.py` and `train_main_glove.py` save **whole modules**
(`torch.save(model, path)`), not `state_dict`s.

## Expected filenames

| File | Produced by | Fusion |
| --- | --- | --- |
| `ConcatEarly.pt` | `train_main_bert.py` | early concat + LSTM head |
| `ConcatLate.pt` | `train_main_bert.py` | late concat |
| `LowRankTensorFusion.pt` | `train_main_bert.py` | LRTF rank 32 |
| `TensorFusion.pt` | `train_main_bert.py` | full tensor product |
| `TransformerEarly.pt` | `train_main_bert.py` | shared transformer, max-pad loader |
| `TransformerLate.pt` | `train_main_bert.py` | per-modality `TransformerSeq` |
| `glove_ConcatEarly.pt` | `train_main_glove.py` | same six names, GloVe widths |
| `glove_ConcatLate.pt` | | |
| `glove_LowRankTensorFusion.pt` | | |
| `glove_TensorFusion.pt` | | |
| `glove_TransformerEarly.pt` | | |
| `glove_TransformerLate.pt` | | |

GMTM weights belong in [`ablation/`](ablation/readme.md).

`.pt` files are gitignored. `model/mosi_test/train_mosi_*.py` load from
this directory with a `../checkpoints/` prefix (run those scripts from
`model/mosi_test/`).

## Loading

```python
import torch
from train_and_test import test

model = torch.load("checkpoints/TransformerLate.pt", map_location="cpu")
# is_packed=True for every fusion except TransformerEarly (and GMTM).
```

The pickle stores a `MultiFramework`, so `models.py` and
`train_and_test.py` must be importable under the same class names.
