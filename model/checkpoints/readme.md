# Main-experiment checkpoints

`train_main_bert.py` and `train_main_glove.py` write full `nn.Module` pickles here:

```
ConcatEarly.pt
ConcatLate.pt
LowRankTensorFusion.pt
TensorFusion.pt
TransformerEarly.pt
TransformerLate.pt
glove_ConcatEarly.pt
glove_ConcatLate.pt
...
```

Weights are **not** stored in git (see the repo `.gitignore`). Recreate them with the train loops in those scripts, or only use `examples/` if you just want shapes and metrics.
