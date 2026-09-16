# Dataset files

The training scripts never open a `.csd` themselves. They load a
**pickle** of already-aligned numpy arrays:

```text
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

Those pickles are large and are not in git (see the root
`.gitignore`). This page is only the *on-disk recipe* for rebuilding
them. The tensor-level description lives in
[docs/datasets.md](../../docs/datasets.md).

## GloVe

`glove.840B.300d.txt` belongs in **this directory**
(`model/data/glove.840B.300d.txt`). The file tracked in git is an
empty placeholder so the path is documented. Replace it with the
official Common-Crawl 840B / 300-d dump from the Stanford GloVe
page before you rebuild a GloVe pickle.

## MOSEI computational sequences

Download from the CMU MOSEI page and place them here:

```text
MOSEI/cmumosei/CMU_MOSEI_COVAREP.csd
MOSEI/cmumosei/CMU_MOSEI_Labels.csd
MOSEI/cmumosei/CMU_MOSEI_TimestampedWords.csd
MOSEI/cmumosei/CMU_MOSEI_VisualFacet42.csd
```

| File | Modality | Width after alignment |
| --- | --- | ---: |
| `CMU_MOSEI_TimestampedWords.csd` | words (then BERT 768 or GloVe 300) | 768 / 300 |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2 | 35 |
| `CMU_MOSEI_COVAREP.csd` | COVAREP | 74 |
| `CMU_MOSEI_Labels.csd` | sentiment `[-3, 3]` | 1 |

`MOSEI/get_dataset.ipynb` is the interactive builder. It still
contains a machine-local `F:\...` path; change that cell first.
`MOSEI/get_mosei.py` is a script-shaped sketch of the same SDK
align + HDF5 dump (the field names inside it are MOSI names — treat
it as a template, not a drop-in MOSEI runner).

## MOSI computational sequences

```text
MOSI/cmumosi/CMU_MOSI_COVAREP.csd
MOSI/cmumosi/CMU_MOSI_Opinion_Labels.csd
MOSI/cmumosi/CMU_MOSI_TimestampedWords.csd
MOSI/cmumosi/CMU_MOSI_Visual_Facet_42.csd
```

Same widths as MOSEI. `MOSI/get_dataset.ipynb` builds
`mosi_raw_bert.pkl` / `mosi_raw_glove.pkl`.

## Expected pickle keys

```python
{
    "train": {"text": ..., "audio": ..., "vision": ..., "labels": ...},
    "valid": {...},
    "test":  {...},
}
```

Each modality is `[N, T, F]`. After `drop_entry` (empty-text clips
removed) the training scripts feed these arrays to `Affectdataset`.

## Do not commit

- the real GloVe dump (~5 GB)
- any `.csd` / `.hdf5` / `.pkl`
- anything you write under `cmumosei/` or `cmumosi/`

Keep this readme and the notebooks; they are the rebuild
instructions.
