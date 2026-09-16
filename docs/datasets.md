# Datasets

Personal notes for the two CMU affect datasets this repo actually trains on. Official distribution is through the [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Nothing from the SDK is checked in here — only the loader code and the expected on-disk layout.

## CMU-MOSI

Short monologue movie-review clips. Each clip has a human sentiment score in **[-3, +3]**.

| Field | Typical value |
| --- | --- |
| Speakers / videos | 93 YouTube reviews |
| Labeled segments | ~2,199 |
| Visual | `CMU_MOSI_Visual_Facet_42` (FACET 4.2) |
| Acoustic | `CMU_MOSI_COVAREP` |
| Text | `CMU_MOSI_TimestampedWords` |
| Labels | `CMU_MOSI_Opinion_Labels` |
| Expected pickle | `model/data/MOSI/mosi_raw_bert.pkl`, `mosi_raw_glove.pkl` |

This repo does **not** train GMTM from scratch on MOSI as the main study. `model/mosi_test/` loads a **MOSEI** checkpoint and scores it on MOSI (sometimes after merging MOSI train/valid/test — see below).

## CMU-MOSEI

The larger in-the-wild set. Same score range, same three families of features.

| Field | Typical value |
| --- | --- |
| Speakers | 1,000+ |
| Labeled sentences | ~23,453 |
| Visual | `CMU_MOSEI_VisualFacet42` |
| Acoustic | `CMU_MOSEI_COVAREP` |
| Text | `CMU_MOSEI_TimestampedWords` |
| Labels | `CMU_MOSEI_Labels` |
| Expected pickle | `model/data/MOSEI/mosei_raw_bert.pkl`, `mosei_raw_glove.pkl` |

MOSEI is the dataset behind `train_main_*.py` and `train_GMTM_*.py`.

## Feature recipe used by the loaders

After alignment and the personal preprocessing notebooks, every clip is a dict of numpy arrays:

```text
{
  "vision": [N, T_raw, 35],
  "audio":  [N, T_raw, 74],
  "text":   [N, T_raw, 768 or 300],
  "labels": [N, 1] or [N, 1, 1],
}
```

stored once per split inside a pickle:

```text
{
  "train": {vision, audio, text, labels},
  "valid": {...},
  "test":  {...},
}
```

`Affectdataset` then:

1. Replaces `-inf` audio with 0 (COVAREP uses `-inf` for unvoiced frames).
2. Optionally trims leading all-zero text (aligned mode) so the three streams start on the first spoken word.
3. Optionally z-normalizes each modality over time (`z_norm=True`; off in the main scripts).
4. Either returns variable-length tensors (packed path) or crops/pads to `max_pad_num=50` (GMTM / early transformer).

`drop_entry` removes clips whose text tensor is all zeros before the DataLoader is built.

## On-disk layout the personal notes expect

From `model/data/readme.md`, expanded:

```text
model/data/
  glove.840B.300d.txt                  # GloVe Common Crawl, not in git
  MOSEI/cmumosei/
    CMU_MOSEI_COVAREP.csd
    CMU_MOSEI_Labels.csd
    CMU_MOSEI_TimestampedWords.csd
    CMU_MOSEI_VisualFacet42.csd
  MOSI/cmumosi/
    CMU_MOSI_COVAREP.csd
    CMU_MOSI_Opinion_Labels.csd
    CMU_MOSI_TimestampedWords.csd
    CMU_MOSI_Visual_Facet_42.csd
```

Download the computational sequences with `mmsdk` (`mmdatasdk.cmu_mosi` / `cmu_mosei`) rather than copying them from another machine. The `.csd` files are large and stay local.

## Alignment (what `get_mosei.py` is doing)

`model/data/MOSEI/get_mosei.py` is a MOSI alignment script despite the folder name. The steps are the usual CMU-SDK pattern:

1. Load words, FACET, COVAREP as a `mmdataset` recipe.
2. `dataset.align(text_field, collapse_functions=[avg])` so visual/audio frames that fall inside a word interval are averaged onto that word.
3. Add opinion labels and `align` again so each labeled segment keeps its temporal word sequence.
4. Dump an HDF5 (`cmumosi/mosi.hdf5`) for later embedding.

The file hard-codes `MOSI_PATH = "F:\\MOSEI\\model\\data\\MOSEI\\cmumosi"`. Point that at your own `cmumosi` directory before running. The Jupyter notebooks under `data/MOSEI/` and `data/MOSI/` are the personal interactive version of the same pipeline (BERT vs GloVe embedding of the word sequence).

Word vectors:

- **GloVe:** look up each timestamped word in `glove.840B.300d.txt`, OOV → zeros.
- **BERT:** encode the word / sentence with a BERT last-layer hidden state (768-d) and align back to the word grid.

## Loader API

All of these live in `model/data/get_dataloader.py`.

| Function | Returns | Notes |
| --- | --- | --- |
| `get_dataloader(path, ...)` | `train, valid, test` | Official splits. Used for MOSEI training. |
| `get_ablation_dataloader(path, modalities=..., embedding=...)` | `train, valid, test` | Zeros unused modalities. |
| `get_mosi_dataloader(path, ...)` | `test` only | **Merges train+valid+test** into one pool. |
| `get_ablation_mosi_dataloader(...)` | `test` only | Same merge, then zero unused modalities. |

Collate functions:

- `_process_1` — pad to the longest item in the batch, also return lengths + ids. Used when `max_pad=False`.
- `_process_2` — stack already-padded length-50 tensors. Used when `max_pad=True`.

Batch that `_process_2` yields (what GMTM / early transformer see):

```text
vision [B, 50, 35], audio [B, 50, 74], text [B, 50, Dt], labels [B, 1]
```

Batch that `_process_1` yields (packed path):

```text
[vision_pad, audio_pad, text_pad], [len_v, len_a, len_t], ids, labels
```

`MultiFramework` with `has_padding=True` consumes the first two of those.

## MOSI transfer merge

`get_mosi_dataloader` concatenates the three official MOSI splits on every key. That is deliberate: the personal MOSI study is “how well does a MOSEI-trained fusion transfer to every MOSI clip?”, not “report MOSI test-only”. If you need a standard MOSI test number, use `get_dataloader` on the MOSI pickle instead.

## Optional z-norm and flattening

`z_norm()` (module-level) and `Affectdataset(z_norm=True)` standardize each clip independently over time. The published CSV runs leave this **off**.

`flatten_time_series=True` collapses `[T, D] → [T*D]` and is unused by the transformer scripts (those need the time axis).

## What is not in this repo

- Raw videos
- `.csd` computational sequences
- GloVe / BERT weights
- Trained `.pt` checkpoints
- The aligned `*_raw_bert.pkl` / `*_raw_glove.pkl` files

The loaders will raise `FileNotFoundError` until you build those pickles locally. The [examples](../examples/README.md) generate synthetic tensors with the same shapes so the model code can be exercised without CMU data.
