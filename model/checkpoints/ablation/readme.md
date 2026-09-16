# Model storage location for the GMTM ablation study

Expected filenames, produced by `train_GMTM_bert.py` / `train_GMTM_glove.py`
when the `train(...)` call is uncommented:

| File | Embedding | Modalities kept (others zeroed) |
| --- | --- | --- |
| `model_text.pt` | BERT | text |
| `model_audio.pt` | BERT | audio |
| `model_visual.pt` | BERT | visual |
| `model_text+audio.pt` | BERT | text+audio |
| `model_text+visual.pt` | BERT | text+visual |
| `model_audio+visual.pt` | BERT | audio+visual |
| `model_text+audio+visual.pt` | BERT | all three |
| `model_glove_text.pt` … `model_glove_text+audio+visual.pt` | GloVe | same seven subsets |

The module is always 3-stream GMTM (`n_modalities=3`, widths `[35, 74, 768]`
or `[35, 74, 300]`). Dropped streams are zero tensors of the official shape,
not missing encoders. See [`docs/data-pipeline.md`](../../../docs/data-pipeline.md).
