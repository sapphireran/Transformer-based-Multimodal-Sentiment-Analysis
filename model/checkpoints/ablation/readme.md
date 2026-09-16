# GMTM ablation checkpoints

Put gated multi-transformer weights here (not committed). Names are
joined modality lists, matching `train_GMTM_*.py`:

```
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
model_glove_text.pt
model_glove_audio.pt
...
model_glove_text+audio+visual.pt
```

The model is always 3-slot GMTM. Dropped modalities are zeros in the
loader, not missing modules. Details: [`docs/datasets.md`](../../../docs/datasets.md).
