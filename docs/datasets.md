# Datasets and on-disk layout

This project consumes **already aligned** MOSI / MOSEI features as Python
pickles. The raw CMU SDK (`.csd`) path is only used by the Windows-era
notebooks under `model/data/MOSEI/` and `model/data/MOSI/`.

## Where files are supposed to live

Documented in the original [`model/data/readme.md`](../model/data/readme.md)
and expanded here.

```
model/data/
  MOSEI/
    mosei_raw_bert.pkl          # word-aligned BERT + FACET + COVAREP
    mosei_raw_glove.pkl         # same, GloVe 840B 300-d text
    cmumosei/
      CMU_MOSEI_COVAREP.csd
      CMU_MOSEI_Labels.csd
      CMU_MOSEI_TimestampedWords.csd
      CMU_MOSEI_VisualFacet42.csd
  MOSI/
    mosi_raw_bert.pkl
    mosi_raw_glove.pkl
    cmumosi/
      CMU_MOSI_COVAREP.csd
      CMU_MOSI_Opinion_Labels.csd
      CMU_MOSI_TimestampedWords.csd
      CMU_MOSI_Visual_Facet_42.csd
  glove.840B.300d.txt           # only needed if you rebuild GloVe pickles
```

None of the pickles, `.csd` files, or GloVe vectors are in git. That is
intentional: they are third-party research data, not this repo's code.

Official processed drops are commonly mirrored from the MultiComp
[processed_data](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
page. The CMU Multimodal SDK is at
[A2Zadeh/CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

## Pickle schema

`get_dataloader.py` does `pickle.load` and expects:

```
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray, "id": list},
  "valid": { ... same keys ... },
  "test":  { ... same keys ... },
}
```

Observed on the original BERT MOSEI pickle (from
`model/data/analysis_dataset.ipynb`):

| Split | Clips (approx.) | Text per clip |
| --- | ---: | --- |
| train | 16327 | `(50, 768)` |
| valid | (standard MOSEI valid) | `(50, 768)` |
| test | (standard MOSEI test) | `(50, 768)` |

`vision` is `(N, T, 35)`, `audio` is `(N, T, 74)`. `labels` is wider than a
single sentiment column (MOSEI ships extra emotion dimensions). The collate
functions take the **first** row after a reshape when `label.shape[1] > 1`,
which is the sentiment intensity used for training.

Ids look like `-3g5yACwYnA[0]` (YouTube id + segment index).

## Cleaning and alignment

`drop_entry` removes clips whose text tensor sums to 0 (empty transcript
after alignment).

`Affectdataset.__getitem__`:

1. Replaces `-inf` in audio with `0` (COVAREP uses `-inf` for unvoiced
   frames in some dumps).
2. If `aligned=True` (default), finds the first non-zero text row and
   trims vision/audio/text from that index — this drops the leading pad
   that some MultiBench-style pickles insert.
3. Optional `z_norm` standardizes each clip over time.
4. Optional `max_pad` truncates to `max_pad_num` (50) and zero-pads.

## Loaders

| Function | Returns | Notes |
| --- | --- | --- |
| `get_dataloader` | `train, valid, test` | Standard split. `raw_path` default is a leftover absolute path and is unused when you pass a pickle. |
| `get_ablation_dataloader` | `train, valid, test` | Same split; unused modalities become zeros |
| `get_mosi_dataloader` | **one** loader | Concatenates MOSI train+valid+test |
| `get_ablation_mosi_dataloader` | **one** loader | Ablation mask + the merged MOSI pool |

`get_mosi_dataloader` merging all three MOSI splits is **not** the official
MOSI test protocol. It was used for a “how does a MOSEI checkpoint behave
on MOSI clips” note. Do not compare those CSVs to papers that report the
standard MOSI test set only.

## Ablation masking

`filter_modalities_list` keeps a 4-slot row `[vision, audio, text, label]`.
Dropped modalities are `torch.zeros(T, F)` with

| Embedding | visual | audio | text |
| --- | --- | --- | --- |
| `bert` | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| `glove` | `(50, 35)` | `(50, 74)` | `(50, 300)` |

GMTM still instantiates three cross-modal towers. A “text-only” run is
therefore “text + two zero streams,” not a smaller network. That is why
text-only GMTM and trimodal GMTM have the **same** parameter count.

## Rebuilding from `.csd`

[`model/data/MOSEI/get_mosei.py`](../model/data/MOSEI/get_mosei.py) is a
sketch that:

1. Loads `CMU_MOSI_*` computational sequences via `mmsdk` (the filename
   says MOSEI; the constants say MOSI — see [quirks](quirks.md)).
2. Aligns visual/audio to words with a mean collapse.
3. Aligns to opinion labels without collapsing time.
4. Writes an HDF5 dump.

The path `F:\MOSEI\model\data\MOSEI\cmumosi` is from the original Windows
box. Rewrite `MOSI_PATH` before running it. The notebooks
`get_dataset.ipynb` contain the interactive version of the same idea.

For everyday work, prefer a known-good pickle over re-deriving alignment.
Word-level alignment choices change every downstream number.

## Synthetic stand-in

[`examples/synthetic_multimodal.py`](../examples/synthetic_multimodal.py)
builds a MOSI-shaped dict (same keys, tiny `N`) so demos and unit tests
can exercise collate logic without CMU data. Labels are drawn in `[-3, 3]`
and optionally correlated with the text channel so a toy trainer has a
signal.
