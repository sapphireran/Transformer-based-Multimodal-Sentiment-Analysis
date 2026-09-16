# Feature and pickle locations

This folder is the **data contract** for the training scripts, not a
host for the actual MOSI / MOSEI files. Those stay local (license +
size). Layout notes and the pickle schema are expanded in
[`../../docs/datasets.md`](../../docs/datasets.md).

## GloVe

Place the official Common Crawl vectors here if you rebuild GloVe
pickles:

```
model/data/glove.840B.300d.txt
```

The file tracked in git is an empty placeholder so the path is
documented. It is **not** the 2 GB vector table.

## MOSEI raw `.csd` (via CMU Multimodal SDK)

Download into `MOSEI/cmumosei/` (or whatever path you pass to `mmsdk`):

| File | Stream |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | audio, 74-D |
| `CMU_MOSEI_Labels.csd` | sentiment |
| `CMU_MOSEI_TimestampedWords.csd` | word intervals |
| `CMU_MOSEI_VisualFacet42.csd` | vision, 35-D |

The training scripts do **not** read these. They read the aligned
pickles:

```
MOSEI/mosei_raw_bert.pkl    # text width 768
MOSEI/mosei_raw_glove.pkl   # text width 300
```

`MOSEI/get_mosei.py` and `MOSEI/get_dataset.ipynb` are the personal
align → pickle trail. `get_mosei.py` still contains a Windows
`F:\MOSEI\...` path from the original machine; change it before
re-running.

## MOSI raw `.csd`

Download into `MOSI/cmumosi/`:

| File | Stream |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | audio, 74-D |
| `CMU_MOSI_Opinion_Labels.csd` | sentiment |
| `CMU_MOSI_TimestampedWords.csd` | word intervals |
| `CMU_MOSI_Visual_Facet_42.csd` | vision, 35-D |

Aligned pickles expected by `model/mosi_test/`:

```
MOSI/mosi_raw_bert.pkl
MOSI/mosi_raw_glove.pkl
```

## What `get_dataloader.py` does

- `drop_entry`: remove clips whose text tensor is all zeros.
- `Affectdataset`: optional z-norm, alignment crop, `-inf` → 0 on audio.
- `get_dataloader`: official train / valid / test loaders.
- `get_ablation_dataloader`: same, with unused modalities zeroed.
- `get_mosi_dataloader`: **concatenates** train+valid+test (transfer
  eval, not the official MOSI test split).

Collate `_process_1` = variable length + lengths (packed LSTM/GRU).
Collate `_process_2` = stacked `[B, 50, F]` (GMTM, TransformerEarly).
