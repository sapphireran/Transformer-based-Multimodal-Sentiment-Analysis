# Data pipeline

Sentiment clips come from **CMU-MOSI** (spoken movie-review snippets) and
**CMU-MOSEI** (YouTube monologues). Both are distributed as CMU Multimodal SDK
computational sequences (`.csd`). This repo does not vendor those files. The
training scripts consume **already-aligned pickle dumps** produced by the
notebooks under `model/data/`.

## On-disk ingredients

From [`model/data/readme.md`](../model/data/readme.md):

| Dataset | File | Role |
| --- | --- | --- |
| GloVe | `glove.840B.300d.txt` | Word vectors if you rebuild text as GloVe |
| MOSEI | `CMU_MOSEI_COVAREP.csd` | Acoustic |
| MOSEI | `CMU_MOSEI_VisualFacet42.csd` | Visual |
| MOSEI | `CMU_MOSEI_TimestampedWords.csd` | Words / alignment spine |
| MOSEI | `CMU_MOSEI_Labels.csd` | Sentiment (and other) labels |
| MOSI | `CMU_MOSI_COVAREP.csd` | Acoustic |
| MOSI | `CMU_MOSI_Visual_Facet_42.csd` | Visual |
| MOSI | `CMU_MOSI_TimestampedWords.csd` | Words |
| MOSI | `CMU_MOSI_Opinion_Labels.csd` | Sentiment labels |

`model/data/MOSEI/get_mosei.py` shows the SDK pattern: load the three feature
sequences, `align` to the word stream with mean collapse, add labels, align
again, optionally dump HDF5. The Windows paths in that file (`F:\MOSEI\...`)
are local to the original machine — rewrite them before rerunning.

The notebooks `get_dataset.ipynb` (MOSI and MOSEI) write the pickles the
loaders expect:

```text
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

## Pickle contract

`analysis_dataset.ipynb` prints the structure. Top-level keys are
`train` / `valid` / `test`. Each split is a dict:

| Key | Type | Typical shape (MOSEI BERT train) |
| --- | --- | --- |
| `vision` | `np.ndarray` | `[N, T, 35]` |
| `audio` | `np.ndarray` | `[N, T, 74]` |
| `text` | `np.ndarray` | `[N, T, 768]` or `[N, T, 300]` |
| `labels` | `np.ndarray` | `[N, 1, K]` — first value is the sentiment score |
| `id` | list of str | clip ids such as `-3g5yACwYnA[0]` |

`T` in the dump is already 50 in the analyzed BERT file. Labels may carry
extra emotion dimensions after the sentiment score; the collate functions keep
the first score (see below).

`drop_entry` removes any clip whose **text** tensor is all zeros. `-inf` in
audio (common in COVAREP unvoiced frames) is replaced with `0.0` inside
`Affectdataset`.

## `Affectdataset`

Constructor flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `flatten_time_series` | caller | If true, returns flat vectors (not used by the recorded scripts) |
| `aligned` | `True` | Slice every stream from the first non-zero text step |
| `max_pad` | `False` | Clip/pad each stream to `max_pad_num` (50) and skip the index column |
| `z_norm` | `False` | Per-clip z-score on each stream |
| `data_type` | `'mosei'` | Only used by the unused `_get_class` helper |

Return values:

- `max_pad=False`: `[vision, audio, text, index, label]`
- `max_pad=True`: `[vision, audio, text, label]` each `[50, F]`

`aligned=True` is why a synthetic text stream in the examples sets
`text[:, 0, 0] = 1` — otherwise `text.nonzero()` is empty and the dataset
`exit()`s.

## Collate functions

Chosen by `max_pad`:

| Function | When | Batch layout |
| --- | --- | --- |
| `_process_1` | `max_pad=False` | `( [v,a,t], [len_v,len_a,len_t], indices, labels )` — sequences padded with `pad_sequence` |
| `_process_2` | `max_pad=True` | `(v, a, t, labels)` — already rectangular `[B, 50, F]` |

Label squeeze: if `label.shape[1] > 1`, the loader reshapes to
`(K, 1)` and takes row `0`. That is how a multi-task label row becomes the
scalar sentiment used by `L1Loss`.

Packed training (`is_packed=True`) consumes `_process_1` and feeds
`[tensors, lengths]` into LSTM/GRU encoders. Transformer-early and GMTM use
`max_pad=True` / `_process_2` so they see dense `[B, 50, F]` stacks.

## Loaders

| Function | Splits returned | Notes |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | Standard MOSEI path. `robust_test` is accepted and ignored. |
| `get_ablation_dataloader` | train, valid, test | After building `Affectdataset`, zero out streams not in `modalities` |
| `get_mosi_dataloader` | **one** loader | Concatenates MOSI train+valid+test, then wraps as `test` |
| `get_ablation_mosi_dataloader` | **one** loader | Same concat, then modality zero-mask |

Ablation zero-mask shapes (after `max_pad=True`):

```text
visual (50, 35)
audio  (50, 74)
text   (50, 768)   if embedding='bert'
text   (50, 300)   if embedding='glove'
```

GMTM always instantiates `n_modalities=3` with those widths. A unimodal
`['text']` run is **not** a 1-stream network; it is a 3-stream network whose
audio and visual inputs are zeros. That is an important reading of the
ablation tables.

## MOSI evaluation is not the official test split

`get_mosi_dataloader` explicitly does:

```python
merged_test_data[key] = np.concatenate([
    alldata['train'][key],
    alldata['valid'][key],
    alldata['test'][key]
], axis=0)
```

So `model/mosi_test/*.csv` scores a MOSEI-trained checkpoint on **all MOSI
clips**. Use those rows as a transfer / domain-shift check. Do not drop them
into a paper table labeled “MOSI test”.

## Batch the training scripts actually feed

MOSEI BERT baselines (`train_main_bert.py`):

- Most methods: `get_dataloader(..., max_pad=False)` → packed LSTM/GRU
- `TransformerEarly`: `get_dataloader(..., max_pad=True)` → dense 50-step stack

GMTM (`train_GMTM_bert.py`):

- `get_ablation_dataloader(..., max_pad=True, embedding='bert')`

Batch size is 32, `num_workers=0`.

## Synthetic stand-in

You do not need the pickles to inspect the contract.
[`examples/dataset_collate_demo.py`](../examples/dataset_collate_demo.py)
builds an 8-clip in-memory dict, runs `Affectdataset` + both collate
functions, and prints batch shapes. That is the same path `get_dataloader`
uses after `pickle.load`.
