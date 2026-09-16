# Main-experiment checkpoints

`train_main_bert.py` and `train_main_glove.py` write one pickle per fusion
method into this directory:

```
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

Each file is a full `torch.save` of a `MultiFramework` (encoders + fusion +
head), not a `state_dict`. The MOSI transfer scripts load these paths
relative to `model/mosi_test/` (`../checkpoints/<name>.pt`).

Weight files are gitignored. Re-run the training scripts to recreate them.
