# Main-experiment checkpoints

Put fusion-sweep weights here (not committed):

| File | Source script |
| --- | --- |
| `ConcatEarly.pt` | `train_main_bert.py` |
| `ConcatLate.pt` | `train_main_bert.py` |
| `LowRankTensorFusion.pt` | `train_main_bert.py` |
| `TensorFusion.pt` | `train_main_bert.py` |
| `TransformerEarly.pt` | `train_main_bert.py` |
| `TransformerLate.pt` | `train_main_bert.py` |
| `glove_{same names}.pt` | `train_main_glove.py` |

`train_mosi_bert.py` currently looks for `Concat.pt` when the method list
says `Concat`. Rename or edit that string if you re-run MOSI transfer.

See [`docs/experiments.md`](../../docs/experiments.md).
