# Checkpoints for GMTM modality ablation

Written by `train_GMTM_bert.py` and `train_GMTM_glove.py`. `*.pt` is
gitignored.

Name pattern:

```
model_{mod}+{mod}+{mod}.pt
model_glove_{mod}+{mod}+{mod}.pt
```

`mod` is one of `text`, `audio`, `visual`, joined with `+` in the order
of the `modality_combinations` list (not sorted). Examples:

- `model_text.pt`
- `model_text+audio.pt`
- `model_text+audio+visual.pt`
- `model_glove_text+audio+visual.pt`

Each file is a full `MultiFramework` pickle whose fusion module is
`GatedMultiTransfomerModel` with **three** towers. Missing modalities were
zero-masked in the loader, not removed from the graph.

MOSI GMTM transfer loads the same files from
`model/mosi_test/../checkpoints/ablation/`.
