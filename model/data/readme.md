# Data files (local, not in git)

Processed pickles and the original CMU computational sequences live on
the machine that ran the sweeps. This folder only keeps the loader and
the alignment scratch notebooks.

## Processed pickles the training scripts expect

Place these next to this readme’s dataset folders:

```
MOSEI/mosei_raw_bert.pkl
MOSEI/mosei_raw_glove.pkl
MOSI/mosi_raw_bert.pkl
MOSI/mosi_raw_glove.pkl
```

Each pickle is `{"train","valid","test"}` →
`{"vision","audio","text","labels"}` as numpy arrays. Widths:

| Key | BERT pickle | GloVe pickle |
| --- | ---: | ---: |
| `vision` | 35 | 35 |
| `audio` | 74 | 74 |
| `text` | 768 | 300 |
| `labels` | 1 (in [-3, 3]) | 1 |

See [`../../docs/datasets.md`](../../docs/datasets.md) for alignment,
`drop_entry`, packed vs. max-pad collate, and the MOSI merge used by
`get_mosi_dataloader`.

## Original CMU `.csd` names

If you rebuild features with [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK):

### GloVe

`glove.840B.300d.txt` — not stored here; too large for git.

### MOSEI (`MOSEI/cmumosei/`)

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

### MOSI (`MOSI/cmumosi/`)

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

`MOSEI/get_mosei.py` is a personal SDK sketch that still hard-codes a
Windows path and, despite the name, aligns **MOSI**. Prefer the notebooks
plus a fresh output path if you regenerate pickles.

## Loader cheat sheet

```python
from data.get_dataloader import get_dataloader, get_ablation_dataloader

train, valid, test = get_dataloader(
    "data/MOSEI/mosei_raw_bert.pkl",
    batch_size=32, data_type="mosei", num_workers=0,
)
# GMTM / TransformerEarly need max_pad=True (T=50, collate _process_2)
train, valid, test = get_dataloader(
    "data/MOSEI/mosei_raw_bert.pkl",
    batch_size=32, max_pad=True, data_type="mosei", num_workers=0,
)
# Zero unused streams instead of dropping them:
train, valid, test = get_ablation_dataloader(
    "data/MOSEI/mosei_raw_bert.pkl",
    embedding="bert", max_pad=True, modalities=["text", "audio"],
    data_type="mosei", num_workers=0,
)
```

Without pickles, use [`../../examples/01_synthetic_batch.py`](../../examples/01_synthetic_batch.py).
