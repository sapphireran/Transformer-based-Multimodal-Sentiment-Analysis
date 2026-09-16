# Datasets and loaders

The experiments use the **word-aligned** CMU-MOSI and CMU-MOSEI feature
packs, not raw video. Nothing in this section downloads data; the large
`.csd` / `.pkl` files stay on the machine that trained the original
checkpoints.

## What each corpus is

| | CMU-MOSI | CMU-MOSEI |
| --- | --- | --- |
| Content | YouTube movie-review clips | YouTube monologues, many topics |
| Label | opinion valence in [-3, 3] | sentiment in [-3, 3] |
| Typical split | ~1.3k / 0.3k / 0.7k segments | ~16k / 1.9k / 4.7k (after dropping empty text) |
| Role in this repo | transfer / stress test (`model/mosi_test`) | main sweeps |

Both corpora share the same three computational sequences after alignment:

| Field (MOSI name) | Field (MOSEI name) | Used as | Width |
| --- | --- | --- | --- |
| `CMU_MOSI_Visual_Facet_42` | `CMU_MOSEI_VisualFacet42` | `vision` | 35 |
| `CMU_MOSI_COVAREP` | `CMU_MOSEI_COVAREP` | `audio` | 74 |
| `CMU_MOSI_TimestampedWords` | `CMU_MOSEI_TimestampedWords` | words → BERT or GloVe | 768 or 300 |
| `CMU_MOSI_Opinion_Labels` | `CMU_MOSEI_Labels` | `labels` | 1 |

`model/data/readme.md` lists the `.csd` filenames the original download
used. GloVe vectors, when rebuilt from words, come from
`glove.840B.300d.txt` (not stored here).

## Expected pickle layout

Training scripts load a single pickle with three dicts:

```
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray},
  "valid": { ... },
  "test":  { ... },
}
```

Each array is `[N, T_raw, F]` (labels often `[N, 1, 1]` or `[N, 1]`).
`T_raw` varies; the dataset class either:

- **packs** variable lengths (`max_pad=False`, collate `_process_1`), or
- **crops / pads to 50** (`max_pad=True`, collate `_process_2`).

Paths hard-coded in the scripts:

```
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

`get_mosei.py` is a leftover CMU-SDK alignment sketch that still points at a
Windows path (`F:\MOSEI\...`) and, despite the filename, builds **MOSI**.
Treat it as a personal scratch script, not the production loader.
`get_dataset.ipynb` in both `MOSEI/` and `MOSI/` is the notebook form of
that alignment (word-average collapse, then align to labels, optional HDF5
dump).

## Loader API

All of these live in `model/data/get_dataloader.py`.

| Function | Returns | Notes |
| --- | --- | --- |
| `get_dataloader` | `train, valid, test` | standard split; `drop_entry` removes empty text |
| `get_ablation_dataloader` | `train, valid, test` | unused modalities become **zeros**, width preserved |
| `get_mosi_dataloader` | `test` only | concatenates train+valid+test into one eval set |
| `get_ablation_mosi_dataloader` | `test` only | same merge, then zero unused modalities |

`drop_entry` deletes an index from **every** key if `text[i].sum() == 0`.

`Affectdataset`:

1. Replaces `-inf` in audio with 0 (COVAREP uses that for unvoiced frames).
2. If `aligned=True` (default), finds the first non-zero text row and
   crops vision / audio / text from that index. This matches the
   word-aligned CMU convention that leading pad is zeros.
3. Optional per-clip z-norm (`z_norm=True`) — **off** in the training
   scripts.
4. `max_pad=True` crops to `max_pad_num` (50) and right-pads with zeros.

Ablation filtering (`filter_modalities_list`) walks the dataset as a list
of rows `[vision, audio, text, label]` and swaps unselected streams for
`torch.zeros(50, F)` with `F` taken from:

```
bert:  visual 35, audio 74, text 768
glove: visual 35, audio 74, text 300
```

That is why GMTM always constructs `n_modalities=3` even for `['text']`.

## Collate functions

`_process_1` (variable length, packed models):

```
[vision_pad, audio_pad, text_pad], [len_v, len_a, len_t], indices, labels
```

`_process_2` (fixed `T=50`):

```
vision, audio, text, labels     # each vision/audio/text is [B, 50, F]
```

`train()` then does `j[:-1]` as the modality list and `j[-1]` as the
target, so `_process_2` batches are exactly what Identity-encoder models
want.

## MOSI “test” merge

`get_mosi_dataloader` **does not** evaluate the official MOSI test split.
It concatenates train, valid, and test. The `mosi_test/` CSVs are therefore
closer to “how does a MOSEI checkpoint score on the whole MOSI pool?” than
to a leaderboard number. That is called out again in the results note.

## Rebuilding features (optional)

If you need to regenerate pickles from CMU `.csd` files:

1. Install [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).
2. Download the computational sequences listed in `model/data/readme.md`.
3. Align words ← Facet / COVAREP with an interval average (the `avg`
   collapse in `get_mosei.py`).
4. Align those sequences to the label computational sequence.
5. Encode words with BERT (`bert-base-uncased` style 768-d) or GloVe 840B.
6. Dump `{train,valid,test}` with the keys above.

The personal examples never touch this path. They allocate
`torch.randn(B, T, F)` with the same `F` values so fusion code can be
exercised without SDK, pickle, or GPU.

## Synthetic stand-in

`examples/common.py` defines `make_aligned_batch` with:

- `VISUAL_DIM = 35`, `AUDIO_DIM = 74`, `BERT_DIM = 768`, `GLOVE_DIM = 300`
- labels uniform in [-3, 3] by default
- `correlated=True` plants a clip-level latent in every text frame and
  sets the label to `3 * tanh(latent)` so `05_mini_training.py` has a
  learnable signal (mean-pooling raw Gaussian text is too weak)

Use that helper whenever a new example needs MOSI-shaped tensors.
