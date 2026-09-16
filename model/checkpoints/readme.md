# Main-experiment checkpoints

`train_main_bert.py` and `train_main_glove.py` call
`torch.save(model, ...)` on the entire `MultiFramework` module (not a
`state_dict`). Drop the resulting files here:

```text
ConcatEarly.pt
ConcatLate.pt
LowRankTensorFusion.pt
TensorFusion.pt
TransformerEarly.pt
TransformerLate.pt
glove_ConcatEarly.pt
glove_ConcatLate.pt
glove_LowRankTensorFusion.pt
glove_TensorFusion.pt
glove_TransformerEarly.pt
glove_TransformerLate.pt
```

These paths are hard-coded in the training and MOSI-transfer
scripts. Reloading requires `models.py` and `train_and_test.py` to
be importable so unpickling can find `MultiFramework` and every
nested class.

Weights are gitignored. This file only reserves the directory.
