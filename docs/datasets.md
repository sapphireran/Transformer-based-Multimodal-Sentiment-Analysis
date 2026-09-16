# Datasets and features

Experiments use two public CMU multimodal affect corpora:

- **CMU-MOSI** — YouTube movie-review clips, utterance-level sentiment.
- **CMU-MOSEI** — larger in-the-wild monologues, same label style.

Labels are continuous sentiment (commonly treated as `[-3, 3]`). This repo
trains with **L1 regression** on that score. Classification numbers are
computed later by binning.

Official project pages (feature lists and licenses live there):

- http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/

Download the computational sequences through the
[CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/force-align-stanfordnlp)
/ `mmsdk` workflow. Do not commit the `.csd` files or derived `.pkl` files;
they are large and redistributed under the dataset terms, not this repo’s
MIT license alone.

## Files this code expects

### Raw computational sequences

Documented in `model/data/readme.md`:

**MOSEI** (`model/data/MOSEI/cmumosei/`)

| File | Modality |
| --- | --- |
| `CMU_MOSEI_VisualFacet42.csd` | visual (Facet 4.2) |
| `CMU_MOSEI_COVAREP.csd` | acoustic |
| `CMU_MOSEI_TimestampedWords.csd` | word timings / tokens |
| `CMU_MOSEI_Labels.csd` | sentiment labels |

**MOSI** (`model/data/MOSI/cmumosi/`)

| File | Modality |
| --- | --- |
| `CMU_MOSI_Visual_Facet_42.csd` | visual |
| `CMU_MOSI_COVAREP.csd` | acoustic |
| `CMU_MOSI_TimestampedWords.csd` | words |
| `CMU_MOSI_Opinion_Labels.csd` | labels |

GloVe vectors, if you rebuild word embeddings yourself:

- `glove.840B.300d.txt` (store path noted in `model/data/readme.md`)

### Processed pickles

| Path | Text encoder | Used by |
| --- | --- | --- |
| `model/data/MOSEI/mosei_raw_bert.pkl` | BERT 768-D | `train_main_bert.py`, `train_GMTM_bert.py` |
| `model/data/MOSEI/mosei_raw_glove.pkl` | GloVe 300-D | `train_main_glove.py`, `train_GMTM_glove.py` |
| `model/data/MOSI/mosi_raw_bert.pkl` | BERT 768-D | `mosi_test/train_mosi_bert.py`, `mult_bert_mosi.py` |
| `model/data/MOSI/mosi_raw_glove.pkl` | GloVe 300-D | GloVe MOSI scripts |

`get_mosei.py` is a **local** MMSDK alignment sketch (hard-coded
`F:\MOSEI\...` paths). Treat it as a personal notebook, not a portable CLI.
The `get_dataset.ipynb` notebooks are the intended pickle builders.

## Pickle schema

`get_dataloader` does `pickle.load` and expects a dict:

```text
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray},
  "valid": { ... },
  "test":  { ... },
}
```

Each array is indexed by utterance:

| Key | Shape | Notes |
| --- | --- | --- |
| `vision` | `(N, T_i, 35)` | Facet features; `-inf` in audio is zeroed in the dataset class |
| `audio` | `(N, T_i, 74)` | COVAREP |
| `text` | `(N, T_i, 768 or 300)` | word-aligned embeddings |
| `labels` | `(N, 1)` or `(N, C)` | if `C > 1`, collate takes the first row after a reshape |

`T_i` may differ across utterances. Packed collate pads to the max length
**in the batch**. Max-pad collate clips/pads to `max_seq_len` (default 50).

## Cleaning steps in the loader

`drop_entry` removes utterances whose **text** tensor sums to 0. All
modalities are deleted at those indices so alignment stays consistent.

`Affectdataset.__getitem__` (aligned mode):

1. Replace `-inf` audio with `0`.
2. Find the first nonzero text row and crop visual/audio/text from there.
   This drops leading pad frames that some preprocessing pipelines insert.
3. Optionally z-normalize each modality over time (`z_norm=True`).
4. Either flatten the whole series, max-pad to 50, or return raw variable
   length plus an index (packed path).

If text is entirely zero after `drop_entry` failed to catch a row, the
aligned branch prints the tensor and `exit()`s. That is a data-quality
assertion, not a recoverable exception.

## Packed vs max-padded collate

`_process_1` (packed, `max_pad=False`):

```text
return (
  [vision_padded, audio_padded, text_padded],  # each (B, t_batch, F)
  [len_v, len_a, len_t],
  indices,                                     # (B, 1)
  labels,                                      # (B, 1)
)
```

`_process_2` (max pad, `max_pad=True`):

```text
return vision, audio, text, labels
# each modality (B, 50, F)
```

`train()` decides how to call the model from `is_packed`. Mismatching the
flag and the collate is the most common local crash
(`list` vs `Tensor` in `LSTM.forward`).

## Ablation loaders

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` keep the three
slots and **zero** dropped modalities:

| Embedding | Zero shape if dropped |
| --- | --- |
| BERT | visual `(50, 35)`, audio `(50, 74)`, text `(50, 768)` |
| GloVe | text becomes `(50, 300)` |

The modality name list uses `visual`, `audio`, `text` (not `vision`). Order
in the row is always visual, audio, text, label.

This is why GMTM always constructs `GatedMultiTransfomerModel(3, ...)`. A
text-only run is still a 3-stream model with silent audio/visual.

## MOSI evaluation loader is not the official split

`get_mosi_dataloader` builds `merged_test_data` by concatenating MOSI
`train`, `valid`, and `test`. The returned object is a **single**
`DataLoader`. Scores in `model/mosi_test/*results.csv` are therefore
“MOSEI checkpoint on all MOSI utterances,” not “MOSI test-set leaderboard.”

MOSEI training uses the normal three-way split via `get_dataloader`.

## Feature-width cheat sheet

| Setup | Visual | Audio | Text | Early-concat `F` |
| --- | --- | --- | --- | --- |
| MOSEI / MOSI BERT | 35 | 74 | 768 | 877 |
| MOSEI / MOSI GloVe | 35 | 74 | 300 | 409 |

Late-concat LSTM widths (BERT): `64 + 256 + 1024 = 1344`.
Late transformer encoder widths (BERT): `64 + 128 + 1024 = 1216`.

## Regenerating pickles (personal machine)

High-level MMSDK flow, matching `get_mosei.py`:

1. Point `MOSI_PATH` / MOSEI path at the directory that holds the `.csd`
   files.
2. Load the visual, acoustic, and word sequences as an `mmdataset`.
3. `align` to words with an averaging collapse so Facet / COVAREP sit on
   the same time grid as tokens.
4. Add the label sequence and align again **without** collapsing, so labeled
   segments keep their temporal structure.
5. Embed words with BERT or GloVe, stack `vision` / `audio` / `text` /
   `labels`, and `pickle.dump` the `{train, valid, test}` dict.

The notebooks perform the BERT vs GloVe split. Re-run them only if you need
new preprocessing; the recorded CSVs already correspond to one such build.

## What is not in git

- `.csd` computational sequences
- `.pkl` processed datasets
- `.pt` trained weights
- `glove.840B.300d.txt`

`examples/synthetic_data.py` builds tensors with the same ranks and dtypes
so documentation examples stay runnable without those files.
