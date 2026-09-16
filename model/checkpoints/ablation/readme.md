# Checkpoints — GMTM ablations

Each file is a `MultiFramework` whose fusion module is
`GatedMultiTransfomerModel` with **three** modality slots. Names list the
streams that were *not* zeroed.

## BERT (`train_GMTM_bert.py`)

| Filename | Active streams |
| --- | --- |
| `model_text.pt` | text |
| `model_audio.pt` | audio |
| `model_visual.pt` | visual |
| `model_text+audio.pt` | text, audio |
| `model_text+visual.pt` | text, visual |
| `model_audio+visual.pt` | audio, visual |
| `model_text+audio+visual.pt` | all three |

## GloVe (`train_GMTM_glove.py`)

Same pattern with a `glove_` prefix:

```text
model_glove_text.pt
model_glove_audio.pt
model_glove_visual.pt
model_glove_text+audio.pt
model_glove_text+visual.pt
model_glove_audio+visual.pt
model_glove_text+audio+visual.pt
```

MOSI transfer scripts load these paths as
`../checkpoints/ablation/model_....pt`.

Tables that go with these files: [`../../../docs/ablation.md`](../../../docs/ablation.md).
