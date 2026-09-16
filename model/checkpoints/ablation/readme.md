# Ablation checkpoints (GMTM)

`train_GMTM_bert.py` / `train_GMTM_glove.py` write one file per
modality subset. The name is `model_` (BERT) or `model_glove_`
plus `'+'.join(modalities)` in the order the list is written
(`text`, `audio`, `visual` — not alphabetical):

```text
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
model_glove_text.pt
...
model_glove_text+audio+visual.pt
```

MOSI transfer scripts load the same filenames from this folder.
Weights are gitignored.
