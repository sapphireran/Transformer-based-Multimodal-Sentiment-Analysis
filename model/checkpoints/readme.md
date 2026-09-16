# Main-experiment checkpoints

`train_main_bert.py` and `train_main_glove.py` save the entire
`MultiFramework` with `torch.save(model, path)` (not a `state_dict`).

| Pattern | Produced by |
| --- | --- |
| `{FusionMethod}.pt` | BERT-MOSEI sweep (`ConcatEarly.pt`, `TransformerLate.pt`, …) |
| `glove_{FusionMethod}.pt` | GloVe-MOSEI sweep |

`FusionMethod` is one of `ConcatEarly`, `ConcatLate`,
`LowRankTensorFusion`, `TensorFusion`, `TransformerEarly`,
`TransformerLate`.

These `.pt` files are gitignored. MOSI transfer scripts load them from
here (`mosi_test/train_mosi_*.py` uses `../checkpoints/...`).

Load on CPU:

```python
model = torch.load("TransformerLate.pt", map_location="cpu")
```

The class definitions in `models.py` / `train_and_test.py` must match the
file. Do not load a BERT-width checkpoint into a GloVe constructor.
