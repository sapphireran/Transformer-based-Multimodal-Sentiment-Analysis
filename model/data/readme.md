# Feature files

This folder holds the aligned multimodal pickles consumed by `get_dataloader.py`.
Raw CMU SDK files and GloVe vectors are **not** committed (they are large and
redistributed by the original dataset owners).

## Expected layout

```
model/data/
  MOSEI/
    mosei_raw_bert.pkl      # vision 35, audio 74, text 768
    mosei_raw_glove.pkl     # vision 35, audio 74, text 300
    cmumosei/               # optional raw CMU-SDK .csd files
  MOSI/
    mosi_raw_bert.pkl
    mosi_raw_glove.pkl
    cmumosi/
  glove.840B.300d.txt       # optional, only needed to rebuild GloVe pickles
```

## GloVe

Download [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) and place
`glove.840B.300d.txt` next to this file if you are rebuilding text embeddings
from the CMU SDK rather than using a pre-built pickle.

## CMU-MOSEI (raw SDK)

Place these computational sequences under `MOSEI/cmumosei/`:

| File | Role |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | acoustic / COVAREP (74-d) |
| `CMU_MOSEI_Labels.csd` | sentiment labels in `[-3, 3]` |
| `CMU_MOSEI_TimestampedWords.csd` | word-aligned text |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2 visual (35-d) |

Rebuild helpers live in `MOSEI/get_dataset.ipynb` and `MOSEI/get_mosei.py`.

## CMU-MOSI (raw SDK)

Place these computational sequences under `MOSI/cmumosi/`:

| File | Role |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | acoustic / COVAREP (74-d) |
| `CMU_MOSI_Opinion_Labels.csd` | sentiment labels in `[-3, 3]` |
| `CMU_MOSI_TimestampedWords.csd` | word-aligned text |
| `CMU_MOSI_Visual_Facet_42.csd` | FACET 4.2 visual (35-d) |

Rebuild helpers live in `MOSI/get_dataset.ipynb`.

## Pickle contract

Each `*_raw_*.pkl` is a dict with `train` / `valid` / `test` splits. Every split
is another dict:

| Key | Shape | Notes |
| --- | --- | --- |
| `vision` | `[N, T, 35]` | FACET 4.2; `-inf` is not used here |
| `audio` | `[N, T, 74]` | COVAREP; `-inf` is zeroed in `Affectdataset` |
| `text` | `[N, T, 768]` or `[N, T, 300]` | BERT or GloVe |
| `labels` | `[N, 1, 1]` or `[N, 1]` | continuous sentiment |

Empty-text rows (`text` sums to 0) are dropped by `drop_entry()` before
batching. Training scripts typically truncate / pad to `T = 50`.

For a walk-through that does **not** need these files, see
[`../../examples/`](../../examples/).
