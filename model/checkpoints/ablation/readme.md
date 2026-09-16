# Checkpoints — GMTM modality ablations

Local only. Full-module pickles (`torch.save(model, path)`).
Class name on the fusion module: `GatedMultiTransfomerModel`.

Scripts: `../../train_GMTM_bert.py`, `../../train_GMTM_glove.py`.
MOSI transfer: `../../mosi_test/mult_bert_mosi.py`, `mult_glove_mosi.py`.

## Filename pattern

Join modalities with `+`, in the order the loop asked for
(`text`, `audio`, `visual` — **not** visual-audio-text):

```text
model_{mods}.pt              # BERT
model_glove_{mods}.pt        # GloVe
```

Expected set:

```text
model_text.pt
model_audio.pt
model_visual.pt
model_text+audio.pt
model_text+visual.pt
model_audio+visual.pt
model_text+audio+visual.pt

model_glove_text.pt
model_glove_audio.pt
model_glove_visual.pt
model_glove_text+audio.pt
model_glove_text+visual.pt
model_glove_audio+visual.pt
model_glove_text+audio+visual.pt
```

Dropped modalities are **zero-filled** at data time; the module is
always 3-way. See [`../../../docs/data-pipeline.md`](../../../docs/data-pipeline.md).

## Logged tables

| Front | MOSEI | MOSI (merged splits) |
| ----- | ----- | -------------------- |
| BERT | `../../ablation_results.csv` | `../../mosi_test/ablation_mosi_results.csv` |
| GloVe | `../../ablation_glove_results.csv` | `../../mosi_test/ablation_mosi_glove_results.csv` |

Write-ups: [`../../../notes/03-gmtm-ablation-mosei.md`](../../../notes/03-gmtm-ablation-mosei.md),
[`../../../notes/04-mosi-transfer.md`](../../../notes/04-mosi-transfer.md).
Architecture: [`../../../notes/05-gmtm-design-notes.md`](../../../notes/05-gmtm-design-notes.md).
