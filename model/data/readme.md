# Data files

Processed pickles are **not** checked in. Rebuild them from the CMU Multimodal
SDK computational sequences, then drop them in the folders below.

## Layout

```
model/data/
  glove.840B.300d.txt          # placeholder in git; replace with the real 840B file
  MOSI/
    get_dataset.ipynb          # builds mosi_raw_bert.pkl / mosi_raw_glove.pkl
    cmumosi/                   # official .csd files (local only)
      CMU_MOSI_COVAREP.csd
      CMU_MOSI_Opinion_Labels.csd
      CMU_MOSI_TimestampedWords.csd
      CMU_MOSI_Visual_Facet_42.csd
    mosi_raw_bert.pkl          # after you run the notebook
    mosi_raw_glove.pkl
  MOSEI/
    get_dataset.ipynb
    get_mosei.py               # older HDF5 export; trainers do not import it
    cmumosei/                  # official .csd files (local only)
      CMU_MOSEI_COVAREP.csd
      CMU_MOSEI_Labels.csd
      CMU_MOSEI_TimestampedWords.csd
      CMU_MOSEI_VisualFacet42.csd
    mosei_raw_bert.pkl
    mosei_raw_glove.pkl
```

## Feature widths

| File | vision | audio | text |
| --- | --- | --- | --- |
| `*_raw_bert.pkl` | 35 | 74 | 768 |
| `*_raw_glove.pkl` | 35 | 74 | 300 |

Sequence length after the notebooks is 50.

## Loaders

`get_dataloader.py` in this folder:

- `get_dataloader` — train / valid / test (use this to **train on MOSI**)
- `get_ablation_dataloader` — same, with unused modalities zeroed
- `get_mosi_dataloader` — **merged** MOSI splits, transfer scoring only
- `get_ablation_mosi_dataloader` — merged + zeroed modalities

Full write-up: [`docs/datasets.md`](../../docs/datasets.md).
MOSI-next plan: [`docs/mosi.md`](../../docs/mosi.md).

A schema-compatible toy pickle (no SDK download):

```bash
python examples/01_synthetic_mosi_dataset.py
```
