# Ablation checkpoints

GMTM weights from `train_GMTM_bert.py` and `train_GMTM_glove.py`.
Not checked in.

| Pattern | Example |
| --- | --- |
| `model_<modalities>.pt` | `model_text+audio+visual.pt` |
| `model_glove_<modalities>.pt` | `model_glove_text+visual.pt` |

`modalities` is `'+'.join(...)` of a subset of `{text, audio, visual}`.

MOSI transfer:

- `mosi_test/mult_bert_mosi.py` → `../checkpoints/ablation/model_<...>.pt`
- `mosi_test/mult_glove_mosi.py` → `../checkpoints/ablation/model_glove_<...>.pt`

Dropped modalities are **zero tensors**, not missing encoders. The graph is
always three-input GMTM.
