# Datasets and features

The training scripts consume **pre-aligned pickle files**, not the raw CMU
SDK streams. Notebooks under `model/data/MOSEI/` and `model/data/MOSI/` show
how those pickles were built from `.csd` computational sequences.

## CMU-MOSEI

[CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) is a
sentence-level collection of YouTube monologues with text, audio, and vision,
labeled for sentiment (and separately for emotions; this repo uses sentiment).

Expected files (see also [`model/data/readme.md`](../model/data/readme.md)):

| File | Role |
| --- | --- |
| `model/data/MOSEI/mosei_raw_bert.pkl` | Aligned BERT features + labels |
| `model/data/MOSEI/mosei_raw_glove.pkl` | Aligned GloVe features + labels |
| `CMU_MOSEI_TimestampedWords.csd` | Word timings for alignment |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2 (35-D vision) |
| `CMU_MOSEI_COVAREP.csd` | COVAREP (74-D audio) |
| `CMU_MOSEI_Labels.csd` | Sentiment labels |

The analysis notebook (`model/data/analysis_dataset.ipynb`) prints a train
split with **16,327** clip ids for the BERT pickle, which matches a standard
MOSEI train partition after dropping empty-text rows.

## CMU-MOSI

[CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) is the
smaller predecessor (2,199 opinion clips). This repo mostly **trains on
MOSEI** and **evaluates the saved weights on MOSI** via
`model/mosi_test/*.py`.

| File | Role |
| --- | --- |
| `model/data/MOSI/mosi_raw_bert.pkl` | BERT features |
| `model/data/MOSI/mosi_raw_glove.pkl` | GloVe features |
| `CMU_MOSI_TimestampedWords.csd` | Word timings |
| `CMU_MOSI_Visual_Facet_42.csd` | FACET 4.2 |
| `CMU_MOSI_COVAREP.csd` | COVAREP |
| `CMU_MOSI_Opinion_Labels.csd` | Opinion / sentiment labels |

`get_mosei.py` (name notwithstanding) is a MOSI MMSDK alignment sketch that
averages features onto word intervals, then onto labeled segments, and can
write `cmumosi/mosi.hdf5`.

## Pickle schema

Each pickle is a `dict` with keys `train`, `valid`, `test`. Every split is
itself a dict:

| Key | Contents |
| --- | --- |
| `vision` | `np.ndarray`, typically `[N, T_raw, 35]` |
| `audio` | `np.ndarray`, `[N, T_raw, 74]` (may contain `-inf`) |
| `text` | `np.ndarray`, `[N, T_raw, 768]` or `[N, T_raw, 300]` |
| `labels` | `np.ndarray`, sentiment; often `[N, 1, 1]` or `[N, 1, K]` |
| `id` | list of clip identifiers, e.g. `-3g5yACwYnA[0]` |

`drop_entry` removes any row whose **text** sums to 0 (empty transcript
features). `-inf` audio values are replaced with `0.0` inside `Affectdataset`.

## `Affectdataset` behavior

Constructor flags used by the scripts:

| Flag | Training default | Effect |
| --- | --- | --- |
| `flatten_time_series` | `False` | If true, each modality is flattened to 1-D. |
| `aligned` | `True` | Slice every modality from the first non-zero **text** frame. |
| `max_pad` | `True` for GMTM / early transformer; `False` otherwise | Truncate to `max_pad_num` and right-pad with zeros. |
| `max_pad_num` | `50` | Sequence length after padding. |
| `z_norm` | `False` in the scripts | Per-sample mean/std normalize each modality. |
| `data_type` | `'mosei'` or `'mosi'` | Only changes the unused `_get_class` helper. |

Unpadded (`max_pad=False`) `__getitem__` returns

```
[vision, audio, text, index, label]
```

Padded returns

```
[vision, audio, text, label]   # each sequence [50, F]
```

## Collate functions

`_process_1` (variable length) pads with `pad_sequence`, returns

```
( [vis, aud, txt], [len_v, len_a, len_t], indices [B,1], labels [B,1] )
```

`_process_2` (fixed length 50) stacks tensors and returns

```
vision, audio, text, labels     # no index tensor
```

`train()` treats `j[:-1]` as model inputs and `j[-1]` as the target, which
matches `_process_2`. Packed training uses the nested list from `_process_1`.

## Modality ablation

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` keep the 3-slot
layout. Modalities **not** listed are replaced with `torch.zeros(T, F)`:

| embedding | visual | audio | text |
| --- | ---: | ---: | ---: |
| `bert` | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| `glove` | `(50, 35)` | `(50, 74)` | `(50, 300)` |

Index map: `visual=0`, `audio=1`, `text=2`. The GMTM therefore always sees
three tensors; a unimodal text run is "text + zeros + zeros", not a 1-input
network. That is why ablation scripts still construct
`GatedMultiTransfomerModel(3, input_dims, ...)`.

Worked example: [`examples/ablation_zeroing.py`](../examples/ablation_zeroing.py).

## MOSI loaders used for transfer

`get_mosi_dataloader` and `get_ablation_mosi_dataloader` **concatenate**
train + valid + test into one evaluation loader (`shuffle=False`). The MOSI
scripts therefore report metrics on the whole MOSI pickle, not only the
canonical test split. Keep that in mind when comparing to papers that score
the 686-clip MOSI test set only.

## GloVe file

`model/data/readme.md` expects `glove.840B.300d.txt` if you rebuild text
features from tokens. The committed experiment CSVs already assume the
pickles exist; the file is not stored in git (see `.gitignore`).

## Synthetic stand-in

[`examples/synthetic_data.py`](../examples/synthetic_data.py) builds the same
ranks (`B×50×35`, `B×50×74`, `B×50×768|300`, labels in `[-3, 3]`) so the
model modules can be exercised without downloading CMU data.
