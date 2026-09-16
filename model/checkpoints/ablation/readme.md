# Ablation-study checkpoints

`train_GMTM_bert.py` / `train_GMTM_glove.py` (when the `train(...)`
block is uncommented) save:

```
model_{modalities}.pt
model_glove_{modalities}.pt
```

`{modalities}` is the `'+'.join(...)` of the kept streams, for
example:

```
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
```

Each file is a `MultiFramework` whose fusion is
`GatedMultiTransfomerModel` and whose encoders/head are `Identity`.
Dropped modalities were **zeroed in the dataloader**, not removed
from the module, so every checkpoint still has three input towers.

These `.pt` files are local-only; git keeps this directory as a
location marker.
