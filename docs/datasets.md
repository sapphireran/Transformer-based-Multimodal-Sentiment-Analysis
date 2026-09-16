# Datasets

Training uses the public academic corpora **CMU-MOSI** and
**CMU-MOSEI**. Raw `.csd` files and the derived `.pkl` caches are **not**
in git (they are large and redistributed under the dataset licenses).
`model/synthetic.py` invents tensors with the same ranks so the
examples keep working offline.

## Expected files

From [`model/data/readme.md`](../model/data/readme.md) and the SDK
scripts:

### GloVe

- `glove.840B.300d.txt` — 300-D Common Crawl vectors, used when
  `embedding='glove'`.

### CMU-MOSEI (`model/data/MOSEI/cmumosei/`)

| File | Role |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | Acoustic (74-D after alignment) |
| `CMU_MOSEI_VisualFacet42.csd` | Facial action / Facet 4.2 (35-D) |
| `CMU_MOSEI_TimestampedWords.csd` | Word timings; alignment target |
| `CMU_MOSEI_Labels.csd` | Continuous sentiment labels |

Derived pickles the scripts actually load:

- `model/data/MOSEI/mosei_raw_bert.pkl`
- `model/data/MOSEI/mosei_raw_glove.pkl`

### CMU-MOSI (`model/data/MOSI/cmumosi/`)

| File | Role |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | Acoustic |
| `CMU_MOSI_Visual_Facet_42.csd` | Facet 4.2 |
| `CMU_MOSI_TimestampedWords.csd` | Words |
| `CMU_MOSI_Opinion_Labels.csd` | Opinion / sentiment labels |

Derived pickles:

- `model/data/MOSI/mosi_raw_bert.pkl`
- `model/data/MOSI/mosi_raw_glove.pkl`

`model/data/MOSEI/get_mosei.py` and the notebooks under
`model/data/MOSEI/` / `model/data/MOSI/` call
[`CMU-MultimodalDataSDK`](https://github.com/A2Zadeh/CMU-MultimodalSDK)
(`mmsdk`). They align visual and acoustic streams to words with a
simple average collapse, then align to the label sequence. An HDF5
dump helper writes `cmumosi/mosi.hdf5` if you want a single-file cache.

Windows-style paths (`F:\MOSEI\...`) in those scripts are leftover from
the original machine. Point `MOSI_PATH` / pickle `filepath` at your
own copies.

## Pickle schema

`get_dataloader` expects a pickle of:

```python
{
  "train": {"text": ndarray, "audio": ndarray, "vision": ndarray, "labels": ndarray},
  "valid": { ... },
  "test":  { ... },
}
```

Ranks after preprocessing:

- `text`  — `[N, T, 768]` (BERT) or `[N, T, 300]` (GloVe)
- `audio` — `[N, T, 74]`
- `vision` — `[N, T, 35]`
- `labels` — `[N, 1, 1]` or `[N, 1]` continuous scores in `[-3, 3]`

`drop_entry` deletes clips whose text sum is identically zero.
`-inf` audio values are replaced with 0. Optional per-clip z-norm is
off unless `z_norm=True`.

## Loader variants

All of these live in [`model/data/get_dataloader.py`](../model/data/get_dataloader.py).

| Function | Splits returned | Notes |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | Standard MOSEI / MOSI loop |
| `get_ablation_dataloader` | train, valid, test | Zeros modalities not in `modalities` |
| `get_mosi_dataloader` | test only | Concatenates train+valid+test into one test set |
| `get_ablation_mosi_dataloader` | test only | Same merge, then modality zeroing |

`max_pad=True` selects `_process_2`: every clip is truncated / padded
to `max_seq_len` (default 50) and the batch is a stacked tensor. This
is required by `TransformerEarly` and GMTM.

`max_pad=False` selects `_process_1`: `pad_sequence` plus a list of
lengths. Packed LSTM / GRU encoders (`has_padding=True`) consume that
form.

Ablation zero-fill widths:

```
visual: (50, 35)
audio:  (50, 74)
text:   (50, 768)   # embedding='bert'
text:   (50, 300)   # embedding='glove'
```

The synthetic helper `zero_out_modalities` copies the same rule.

## Labels

MOSI and MOSEI annotate each clip with a real-valued opinion score,
commonly treated as:

- **regression** target in `[-3, 3]` (this repo uses `L1Loss`)
- **binary** positive / negative after dropping zeros
- **5-class / 7-class** after uniform binning of `[-3, 3]`

See [evaluation.md](evaluation.md).

## Why examples do not ship pickles

The SDK archives are hundreds of megabytes and the BERT pickle is
larger still. Checking them in would also step on the dataset
licenses. `examples/` therefore builds `SyntheticBatch` objects: same
`[B, T, F]` ranks, labels in `[-3, 3]`, optional text-derived signal
so a tiny GMTM can overfit in a few steps.
