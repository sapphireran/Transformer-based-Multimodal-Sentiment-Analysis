# Checkpoints — GMTM and modality ablations

Written by `train_GMTM_bert.py` / `train_GMTM_glove.py`. The fusion slot is
a `GatedMultiTransfomerModel`; encoders and the head are `Identity`.
Missing modalities were **zero tensors**, so every file still has three
input slots (`n_modalities=3`).

| File | Embedding | Kept modalities |
| --- | --- | --- |
| `model_text.pt` | BERT | text |
| `model_audio.pt` | BERT | audio |
| `model_visual.pt` | BERT | visual |
| `model_text+audio.pt` | BERT | text + audio |
| `model_text+visual.pt` | BERT | text + visual |
| `model_audio+visual.pt` | BERT | audio + visual |
| `model_text+audio+visual.pt` | BERT | all three |
| `model_glove_text.pt` | GloVe | text |
| `model_glove_audio.pt` | GloVe | audio |
| `model_glove_visual.pt` | GloVe | visual |
| `model_glove_text+audio.pt` | GloVe | text + audio |
| `model_glove_text+visual.pt` | GloVe | text + visual |
| `model_glove_audio+visual.pt` | GloVe | audio + visual |
| `model_glove_text+audio+visual.pt` | GloVe | all three |

Recorded metrics for these runs: `model/results/ablation_results.csv` and
`ablation_glove_results.csv`. MOSI transfer of the same files is logged
under `model/mosi_test/`.
