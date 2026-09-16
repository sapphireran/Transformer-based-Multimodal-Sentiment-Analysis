# Datasets and the pickle contract

Training scripts never read the raw `.csd` files. They read **aligned
pickles** produced from CMU-MOSI / CMU-MOSEI. This page is the contract
those pickles must satisfy, plus how `Affectdataset` massages them.

## Official sources

| Corpus | What it is | Typical size | Labels |
| --- | --- | --- | --- |
| [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | YouTube movie-review clips, one speaker, short opinions | ~2.2k segments | human sentiment ≈ `[-3, 3]` |
| [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | In-the-wild monologues, many topics | ~23k segments | same sentiment scale, plus extra emotion annotations this repo does not use |

Both ship through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).
The notebooks under `model/data/MOSEI/` and `model/data/MOSI/` are a
personal record of that download + word-align step.

## Raw computational sequences this repo used

Paths in `model/data/readme.md` and `model/data/MOSEI/get_mosei.py`
(the latter file is named MOSEI but the constants still say MOSI —
copy-paste leftover):

### MOSI (`model/data/MOSI/`)

| File | Stream |
| --- | --- |
| `CMU_MOSI_Visual_Facet_42.csd` | FACET 4.2 facial action units, **35-D** |
| `CMU_MOSI_COVAREP.csd` | COVAREP acoustics, **74-D** |
| `CMU_MOSI_TimestampedWords.csd` | word intervals for alignment |
| `CMU_MOSI_Opinion_Labels.csd` | segment sentiment |

### MOSEI (`model/data/MOSEI/`)

| File | Stream |
| --- | --- |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2, **35-D** |
| `CMU_MOSEI_COVAREP.csd` | COVAREP, **74-D** |
| `CMU_MOSEI_TimestampedWords.csd` | word intervals |
| `CMU_MOSEI_Labels.csd` | segment sentiment |

GloVe text is `glove.840B.300d.txt` (the checked-in file is an empty
placeholder). BERT text is the 768-D word-aligned encoding written into
the pickle by the `get_dataset.ipynb` notebooks.

`get_mosei.py` aligns every stream to words with a mean collapse, then
aligns those to the label intervals, then dumps HDF5. The notebooks
turn that into the pickle the dataloaders expect.

## Pickle layout

`get_dataloader.py` does `pickle.load` and expects:

```
{
  "train": {"text": ndarray, "audio": ndarray, "vision": ndarray, "labels": ndarray, ...},
  "valid": { ... },
  "test":  { ... },
}
```

Convention used throughout this repo:

| Key | Shape | Notes |
| --- | --- | --- |
| `text` | `[N, T, 768]` or `[N, T, 300]` | BERT vs GloVe |
| `audio` | `[N, T, 74]` | `-inf` is replaced with `0` in `Affectdataset` |
| `vision` | `[N, T, 35]` | FACET |
| `labels` | `[N, 1, 1]` or `[N, 1]` | continuous sentiment |

`analysis_dataset.ipynb` prints `valid['text'][0].shape == (50, 768)`
for the BERT MOSEI pickle that was on disk when the notebook ran, so
the working personal copy was already truncated to **T = 50**.

`drop_entry` deletes any clip whose **text** tensor is all zeros
(empty transcript). Audio / vision / labels are deleted on the same
indices.

## Expected on-disk names

The training scripts hard-code these relative paths (cwd = `model/`):

```
data/MOSEI/mosei_raw_bert.pkl
data/MOSEI/mosei_raw_glove.pkl
data/MOSI/mosi_raw_bert.pkl      # used from model/mosi_test/ as ../data/MOSI/...
data/MOSI/mosi_raw_glove.pkl
```

Those pickles are **not** in git (they are large and redistributable
only under the CMU licenses). The empty `glove.840B.300d.txt` stub is
only a reminder of where the 300-D vectors should sit.

## `Affectdataset` behaviour

Constructor flags that matter:

| Flag | Effect |
| --- | --- |
| `aligned=True` (default) | Crop all three streams from the first non-zero **text** frame |
| `aligned=False` | Crop each stream from its own first non-zero frame |
| `z_norm=True` | Per-clip, per-feature z-score; NaNs → 0 |
| `flatten_time_series=True` | Return flat vectors (unused by current scripts) |
| `max_pad=True` | Truncate to `max_pad_num` (50) and zero-pad |
| `max_pad=False` | Variable-length sequences; collate with `pad_sequence` |
| `task` / `data_type` | Only used by the unused `_get_class` helper |

Audio `-inf` (common in COVAREP log-pitch) is zeroed in `__init__`.

Two collate functions:

- `_process_1` (`max_pad=False`): returns
  `[vision, audio, text], lengths, ids, labels`.
  Packed LSTM/GRU baselines use this (`is_packed=True`).
- `_process_2` (`max_pad=True`): returns
  `vision, audio, text, labels` as stacked `[B, 50, F]` tensors.
  GMTM and TransformerEarly use this.

Labels with a trailing extra axis are reshaped so the trainer always
sees `[B, 1]`.

## Ablation masking

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` keep the
three-tensor API and replace dropped modalities with zeros of a fixed
shape:

```
visual: (50, 35)
audio:  (50, 74)
text:   (50, 768) if embedding == "bert" else (50, 300)
```

The modality name list is `['visual', 'audio', 'text']` internally, but
the training scripts pass `'text' / 'audio' / 'visual'`. Those strings
must match the keys in `modality_index_map`.

**MOSI eval quirk:** `get_mosi_dataloader` and
`get_ablation_mosi_dataloader` **concatenate train+valid+test** into
one loader named `test`. That is intentional for the "train on MOSEI,
look at MOSI" notebooks, but it is **not** the official MOSI test
split. Numbers in `model/mosi_test/*.csv` should be read as
*transfer / pooled MOSI*, not as leaderboard MOSI test.

## Z-normalization helper

`z_norm(dataset, max_seq_len=50)` in `get_dataloader.py` is a numpy
preprocessing path that is **not** called by `get_dataloader`. The
dataset class does the same math when `z_norm=True`. Main scripts leave
that flag off.

## Building a pickle from scratch (personal checklist)

1. Install `mmsdk` and download the four `.csd` files per corpus into
   `model/data/{MOSI,MOSEI}/cmu{mosi,mosei}/`.
2. Align to words, then to labels (see `get_mosei.py`).
3. Replace the word tokens with GloVe-840B or a frozen BERT word
   encoding. The notebooks under `data/MOSEI/get_dataset.ipynb` and
   `data/MOSI/get_dataset.ipynb` are the personal trail for that step.
4. `pickle.dump` a dict with `train` / `valid` / `test` and the four
   arrays above. Use the standard CMU-MOSEI / MOSI split keys.
5. Smoke-test with `examples/inspect_pickle.py` if you add a local
   pickle (the example skips cleanly when the file is absent).

## Synthetic stand-in

`examples/synthetic_batch.py` builds tensors with the same ranks and
default widths so architecture code can be exercised without the CMU
license or a multi-GB download.
