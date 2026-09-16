# Main-experiment checkpoints

`torch.save(MultiFramework, path)` files produced by the MOSEI fusion trainers.
They are not checked in (see the root `.gitignore`).

| Trainer | Files written here |
| --- | --- |
| `train_main_bert.py` | `ConcatEarly.pt`, `ConcatLate.pt`, `LowRankTensorFusion.pt`, `TensorFusion.pt`, `TransformerEarly.pt`, `TransformerLate.pt` |
| `train_main_glove.py` | `glove_<same names>.pt` |

MOSI transfer scripts load these from `model/mosi_test/` as `../checkpoints/<name>.pt`.

Watch the `Concat` vs `ConcatLate` name: `train_mosi_bert.py` asks for `Concat.pt`.
Either symlink or copy `ConcatLate.pt` before that script.

GMTM ablation weights go in [`ablation/`](ablation/readme.md).
A future MOSI-native run should use `model/checkpoints/mosi/`
(see [`docs/mosi.md`](../../docs/mosi.md)).
