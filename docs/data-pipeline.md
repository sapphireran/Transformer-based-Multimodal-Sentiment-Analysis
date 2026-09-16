# Data pipeline

The training loop never reads raw video. Everything goes through a
pickle that already holds aligned `vision`, `audio`, `text`, and
`labels` arrays for `train` / `valid` / `test`.

## Expected files

Documented in [`model/data/readme.md`](../model/data/readme.md):

| File | Used by |
| --- | --- |
| `model/data/MOSEI/mosei_raw_bert.pkl` | `train_main_bert.py`, `train_GMTM_bert.py` |
| `model/data/MOSEI/mosei_raw_glove.pkl` | `train_main_glove.py`, `train_GMTM_glove.py` |
| `model/data/MOSI/mosi_raw_bert.pkl` | `model/mosi_test/train_mosi_bert.py`, `mult_bert_mosi.py` |
| `model/data/MOSI/mosi_raw_glove.pkl` | `model/mosi_test/train_mosi_glove.py` |

Raw CMU-SDK computational sequences (not committed):

**MOSEI** under `MOSEI/cmumosei/`

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

**MOSI** under `MOSI/cmumosi/`

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

GloVe vectors: `glove.840B.300d.txt` (not committed). BERT features were
pre-extracted into the pickle; there is no live Hugging Face call in the
training scripts.

## Pickle schema

`pickle.load` returns a dict of splits. Each split is a dict of numpy
arrays with a shared leading sample axis:

```
alldata['train']['vision']  -> (N, T, 35)
alldata['train']['audio']   -> (N, T, 74)
alldata['train']['text']    -> (N, T, 768 or 300)
alldata['train']['labels']  -> (N, 1, 1) or (N, 1)
```

`T` in the raw pickle can be longer than 50. The dataset class trims
and / or left-aligns from the first non-zero text row.

`drop_entry` removes any sample whose **text** tensor sums to 0. The
same indices are deleted from every modality, including labels. Empty
transcripts therefore never reach the model.

## `Affectdataset.__getitem__`

1. Convert vision / audio / text to tensors.
2. If `aligned=True` (default), find the first non-zero text timestep
   and slice all three streams from that index. If `aligned=False`,
   each stream is sliced from its own first non-zero row.
3. Optionally z-normalize each stream independently
   `(x - mean_t) / std_t`, replacing NaNs with 0.
4. Replace `-inf` in audio with 0 at init time.
5. Return either:
   - padded: `[vision, audio, text, label]` with each sequence cut to
     `max_pad_num` (50) and zero-padded on the time axis
   - unpacked: `[vision, audio, text, index, label]` with variable `T`

Labels stay as floats. Binary class mapping exists in `_get_class` but
is unused by the current `__getitem__`.

## Two collate functions

### `_process_1` — packed / variable length

Used when `max_pad=False`. Pads each modality independently with
`pad_sequence`, records lengths, and returns

```
(features_list, lengths_list, indices, labels)
```

`features_list[i]` is `(B, T_max_i, F_i)`. This is the path for
ConcatEarly / ConcatLate / TensorFusion / LRF / TransformerLate, which
set `is_packed=True` so `MultiFramework` feeds `[tensor, length]` into
each encoder.

### `_process_2` — fixed pad

Used when `max_pad=True`. Stacks already-padded sequences and returns

```
vision, audio, text, labels
```

each vision/audio/text tensor `(B, 50, F)`. This is the path for
TransformerEarly and GMTM (`is_packed=False`).

The same pickle can be loaded both ways. `train_main_bert.py` builds
two dataloader triples (`*_OT` packed, `*_TE` padded) and swaps them
per fusion method.

## Ablation loaders

`get_ablation_dataloader` and `get_ablation_mosi_dataloader` keep the
three-slot layout even when a run is "audio only". Missing modalities
are replaced with **zeros of the official shape**:

| embedding | visual | audio | text |
| --- | --- | --- | --- |
| `bert` | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| `glove` | `(50, 35)` | `(50, 74)` | `(50, 300)` |

GMTM therefore always sees `n_modalities=3` and the same
`input_dims`. A unimodal run is a zero-ablation, not a smaller
network. That is why `train_GMTM_*.py` hard-codes
`GatedMultiTransfomerModel(3, input_dims, ...)`.

## MOSI evaluation loaders merge splits

`get_mosi_dataloader` and `get_ablation_mosi_dataloader` concatenate
`train`, `valid`, and `test` before building the **test** loader.
MOSEI loaders do **not** do this.

I left the merge in place because the MOSI numbers in
`model/mosi_test/*.csv` were produced that way. If you want a clean
held-out MOSI score, build a loader from `alldata['test']` only.
[`examples/packed_vs_padded.py`](../examples/packed_vs_padded.py)
shows both collate paths on synthetic clips so you can see the return
signatures without a pickle.

## Alignment recipe (raw SDK)

[`model/data/MOSEI/get_mosei.py`](../model/data/MOSEI/get_mosei.py) is a
CMU-SDK sketch (the constants still say MOSI). The idea:

1. Load word, Facet, and COVAREP computational sequences.
2. `dataset.align(text_field, collapse_functions=[avg])` so visual and
   acoustic frames collapse onto word intervals.
3. Add opinion / sentiment labels and align again, this time **without**
   collapsing, so labeled segments keep their temporal length.
4. Optionally dump an HDF5.

The notebooks under `model/data/MOSI/` and `model/data/MOSEI/` are the
interactive version of the same recipe. They are not required once the
pickles exist.

## Normalization choices

- Loader-level `z_norm=False` on every recorded script.
- Audio `-inf` → `0` always.
- No global (dataset-wide) mean/std; if you turn `z_norm` on, it is
  per-sample, per-feature.

## Synthetic stand-in

When you do not have the pickles,
[`examples/synthetic_data.py`](../examples/synthetic_data.py) builds
dict-of-arrays with the same keys and dtypes, plus helpers that mimic
`_process_1` and `_process_2`. The toy trainer uses that instead of
`Affectdataset`.
