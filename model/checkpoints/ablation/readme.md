# GMTM ablation checkpoints

`train_GMTM_bert.py` / `train_GMTM_glove.py` save:

```
model_{text+audio+visual}.pt
model_glove_{text}.pt
model_glove_{audio}.pt
...
```

The `{...}` part is `'+'.join(modalities)` from the ablation list. MOSI testers under `model/mosi_test/` load these with a `../checkpoints/ablation/` prefix.

Not committed. See `docs/ablation.md`.
