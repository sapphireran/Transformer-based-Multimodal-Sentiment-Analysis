# GMTM ablation checkpoints

`train_GMTM_bert.py` and `train_GMTM_glove.py` save one gated
multi-transformer per modality subset:

```
model_<mod>+<mod>.pt          # BERT, e.g. model_text+audio+visual.pt
model_glove_<mod>+<mod>.pt    # GloVe
```

Subsets match `examples/ablate_toy_modalities.py`:

```
text
audio
visual
text+audio
text+visual
audio+visual
text+audio+visual
```

Dropped modalities are **zero tensors**, not missing modules, so every file
is a 3-input GMTM. MOSI transfer loads them from
`model/mosi_test/../checkpoints/ablation/`.
