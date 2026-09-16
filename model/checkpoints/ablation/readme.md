# Checkpoints for GMTM modality ablations

`train_GMTM_bert.py` and `train_GMTM_glove.py` write here. Names are the
modality list joined with `+`. Unused streams were **zero-filled**, not
removed, so every file is still a 3-input `GatedMultiTransfomerModel`.

## BERT (`input_dims = [35, 74, 768]`)

```
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
```

## GloVe (`input_dims = [35, 74, 300]`)

```
model_glove_text.pt
model_glove_audio.pt
model_glove_visual.pt
model_glove_text+audio.pt
model_glove_text+visual.pt
model_glove_audio+visual.pt
model_glove_text+audio+visual.pt
```

`mosi_test/mult_bert_mosi.py` and `mult_glove_mosi.py` load these paths
with a `../checkpoints/ablation/` prefix.

HParams baked into the trained modules (see those scripts):

```
embed_dim=64, num_heads=4, layers=4
attn_dropout_modalities=[0, 0, 0.1]
embed_dropout=0.2, out_dropout=0.1
```

A CPU-sized cousin (`embed_dim=16`, 2 layers) is what
[`../../../examples/03_gmtm_forward.py`](../../../examples/03_gmtm_forward.py)
constructs from scratch — it does not load a file from this folder.
