# Checkpoints (GMTM ablations)

Directory for `train_GMTM_bert.py` / `train_GMTM_glove.py`.

Expected names:

```
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
model_glove_text.pt
...
```

Each file is a full `MultiFramework` whose fusion module is
`GatedMultiTransfomerModel` and whose encoders / head are `Identity`.
The class name is spelled `GatedMultiTransfomerModel` (missing 'o');
do not rename it if you still want to unpickle these files.

`.pt` files are gitignored. MOSI transfer scripts load the BERT
checkpoints from `../checkpoints/` and the ablation ones from this
folder.
