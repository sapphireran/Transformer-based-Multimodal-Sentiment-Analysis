# Checkpoints (main bake-off)

Directory for the six generic fusion graphs trained by
`train_main_bert.py` and `train_main_glove.py`.

Expected names after a full BERT run:

```
ConcatEarly.pt
ConcatLate.pt
LowRankTensorFusion.pt
TensorFusion.pt
TransformerEarly.pt
TransformerLate.pt
```

GloVe twins are prefixed `glove_` (`train_main_glove.py`).

`.pt` files are gitignored. `torch.load` of the historical files
unpickles a whole `MultiFramework` (encoders + fusion + head), not a
`state_dict`. Retrain before evaluating `mosi_test/train_mosi_*.py`.

See [`docs/training.md`](../../docs/training.md) and
[`docs/reproducibility.md`](../../docs/reproducibility.md).
