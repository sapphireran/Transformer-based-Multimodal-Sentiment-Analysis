# Checkpoints — GMTM / modality ablation

`train_GMTM_bert.py` and `train_GMTM_glove.py` write whole-model pickles here. Gitignored.

BERT (`input_dims = [35, 74, 768]`):

```text
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt
```

GloVe (`input_dims = [35, 74, 300]`), same stems with a `model_glove_` prefix:

```text
model_glove_text.pt
model_glove_audio.pt
model_glove_visual.pt
model_glove_text+audio.pt
model_glove_text+visual.pt
model_glove_audio+visual.pt
model_glove_text+audio+visual.pt
```

The architecture is always three towers. A “text-only” file is a full GMTM trained with audio and visual inputs zeroed, not a single-encoder net. See [docs/architecture.md](../../../docs/architecture.md) and [docs/training.md](../../../docs/training.md).
