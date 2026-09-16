# Main-experiment checkpoints

`train_main_bert.py` and `train_main_glove.py` write one full-module
pickle per fusion name:

```
ConcatEarly.pt
ConcatLate.pt
LowRankTensorFusion.pt
TensorFusion.pt
TransformerEarly.pt
TransformerLate.pt
```

These files are **not** stored in git. They are `torch.save(model)`
dumps of a `MultiFramework` instance and are specific to the machine
and PyTorch version that produced them.

Load example (PyTorch 2.6+ needs `weights_only=False`):

```python
import torch
model = torch.load("ConcatLate.pt", map_location="cpu", weights_only=False)
model.eval()
```

The MOSI transfer scripts in `../mosi_test/` look for the BERT/GloVe
main checkpoints at `../checkpoints/{name}.pt`. Some MOSI scripts use
the filename `Concat.pt` for late concat — keep a symlink or copy if
you need those rows.
