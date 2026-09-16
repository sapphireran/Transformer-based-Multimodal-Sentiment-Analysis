# Datasets and features

This project does not stream raw video. Training always reads a **pickle**
with pre-aligned, per-timestep features. The pickles themselves are not in
git; this page describes the schema the loaders already expect.

## Which corpora

| Corpus | Used as | Scripts |
| --- | --- | --- |
| [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | train / valid / test | `train_main_*.py`, `train_GMTM_*.py` |
| [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | test-only transfer | `model/mosi_test/*.py` |

MOSEI is the larger in-the-wild monologue set. MOSI is the older, smaller
movie-review set. Checkpoints are trained on MOSEI and, in `mosi_test/`,
evaluated on MOSI **without** a MOSI training loop (the fusion scripts
there only call `test()`).

Both corpora annotate sentiment on a roughly `[-3, +3]` Likert-style scale.
This repo treats that number as a regression target.

## On-disk raw files (MMSDK)

Listed in `model/data/readme.md`. After you download the CMU SDK dumps:

**MOSEI** (`model/data/MOSEI/cmumosei/`)

- `CMU_MOSEI_COVAREP.csd` — audio
- `CMU_MOSEI_VisualFacet42.csd` — vision
- `CMU_MOSEI_TimestampedWords.csd` — words / alignment spine
- `CMU_MOSEI_Labels.csd` — sentiment labels

**MOSI** (`model/data/MOSI/cmumosi/`)

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Visual_Facet_42.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Opinion_Labels.csd`

GloVe vectors, if you rebuild text from words rather than using the BERT
pickle: `glove.840B.300d.txt` (location left to you; the data readme only
marks the heading).

`model/data/MOSEI/get_mosei.py` shows the MMSDK alignment recipe used on the
original machine. Paths in that file are Windows-specific
(`F:\MOSEI\...`) and the script name says MOSEI while the constants point at
`cmu_mosi` — treat it as a personal scratch script, not a portable CLI.
The notebooks `get_dataset.ipynb` (under both `MOSEI/` and `MOSI/`) are the
interactive versions of the same idea: read `.csd` sequences, align to
words, align to labels, dump arrays.

## Pickle schema the training code actually loads

`get_dataloader.py` does:

```python
alldata = pickle.load(f)          # dict
alldata['train'] / 'valid' / 'test'
```

Each split is a dict with NumPy arrays (and an optional id list):

| Key | Shape (typical) | Notes |
| --- | --- | --- |
| `vision` | `(N, T_raw, 35)` | FACET 4.2 |
| `audio` | `(N, T_raw, 74)` | COVAREP; `-inf` is replaced with 0.0 |
| `text` | `(N, T_raw, 768)` or `(N, T_raw, 300)` | BERT or GloVe |
| `labels` | `(N, 1, …)` or `(N, 1)` | first column is the sentiment score when extra heads exist |
| `id` | list of `N` strings | e.g. `-3g5yACwYnA[0]` (present in the analysis notebook) |

`analysis_dataset.ipynb` recorded a BERT-MOSEI train split of **16327**
utterances and printed `text` rows of length `(50, 768)` after truncation.

Expected filenames, hard-coded in the train scripts:

```
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

## What `Affectdataset` does per row

`model/data/get_dataloader.py`, class `Affectdataset`:

1. **Drop empty text** happens *before* the Dataset is built, in
   `drop_entry()`: any row whose text tensor sums to 0 is removed from
   every modality, including labels.
2. **Leading-pad strip (aligned=True, the default).** Finds the first
   non-zero entry of the text sequence and slices vision / audio / text
   from that index. Unaligned mode slices each modality from its own first
   non-zero.
3. **Optional z-norm** per utterance, per feature, with `nan_to_num`.
   Off by default in every train script (`z_norm=False`).
4. **Flatten** (`flatten_time_series=True`) would return one long vector
   per modality. Unused by the published scripts.
5. **`max_pad=True`** truncates each modality to `max_pad_num` (50) and
   zero-pads on the time axis so every sample is `(50, F)`. This is the
   TransformerEarly / GMTM path.
6. **Label tensor** is `float`. Binary helper `_get_class` exists but the
   returned label is still the raw score.

Audio `-inf` values are zeroed in `__init__` for the whole split.

## Two collate functions

### `_process_1` — variable length (packed)

Used when `max_pad=False`. Each sample is
`[vision, audio, text, index, label]`.

The collate returns:

```
(processed_input,      # list of 3 padded (B, T_max, F_i) tensors
 processed_lengths,    # list of 3 length tensors
 indices,              # (B, 1)
 labels)               # (B, 1)
```

`train(..., is_packed=True)` then feeds
`[[vision, audio, text], lengths]` into `MultiFramework`.

If `label` has more than one column, only the first is kept
(`reshape(...)[0]`).

### `_process_2` — fixed length (max-pad)

Used when `max_pad=True`. Each sample is `[vision, audio, text, label]`
(no index). Collate stacks to:

```
vision, audio, text, labels   # each batched, T=50
```

`train(..., is_packed=False)` feeds `[vision, audio, text]`.

## Ablation zero-out

`get_ablation_dataloader` and `get_ablation_mosi_dataloader` do **not**
rebuild a 1- or 2-modality Dataset. They walk every row and, for each
modality **not** in the requested list, replace the tensor with zeros of
the official shape:

| Embedding | vision | audio | text |
| --- | --- | --- | --- |
| BERT | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| GloVe | `(50, 35)` | `(50, 74)` | `(50, 300)` |

So GMTM always has `n_modalities=3` and `n_features=[35, 74, 768|300]`.
A `['text']` run is “text plus silent audio/vision”, not a smaller
network. That is why you can load `model_text.pt` and
`model_text+audio+visual.pt` with the same class constructor.

`get_mosi_dataloader` / `get_ablation_mosi_dataloader` additionally
**concatenate train+valid+test** of the MOSI pickle into one test loader.
The comment in the file says this is intentional: MOSI is used as a single
evaluation pool for a MOSEI-trained checkpoint.

## Synthetic stand-in

`examples/synthetic_affect.py` builds a dict with the same keys and
shapes (`N` small, `T=50`, the four widths above) plus labels in
`[-3, 3]`. It is enough to exercise GMTM and the metric helpers. It is
**not** a substitute for reporting numbers: do not compare toy MAE to the
CSVs in `model/results/`.

## Feature widths, one more time

These constants are repeated in every train script for a reason — they
must match the pickle you load.

```
VISUAL_DIM = 35
AUDIO_DIM  = 74
BERT_DIM   = 768
GLOVE_DIM  = 300
MAX_LEN    = 50
```

`examples/configs/*.yaml` stores the same numbers next to optimizer
settings so you do not have to grep `train_main_bert.py`.
