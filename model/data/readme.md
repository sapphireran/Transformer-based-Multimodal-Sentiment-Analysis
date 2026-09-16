# Feature files

The training scripts never read the raw `.csd` streams. They load **aligned
pickle files** produced from those streams (notebooks in `MOSEI/` and `MOSI/`).

Longer notes: [`docs/datasets.md`](../../docs/datasets.md).

## GloVe (only if you rebuild text features)

Place `glove.840B.300d.txt` somewhere local and point the notebook at it.
The file is gitignored (~2 GB). Committed GloVe *results* already assume a
finished `mosei_raw_glove.pkl` / `mosi_raw_glove.pkl`.

## MOSEI computational sequences

Put these under `MOSEI/cmumosei/`:

| File | Modality |
| --- | --- |
| `CMU_MOSEI_TimestampedWords.csd` | text timings (alignment target) |
| `CMU_MOSEI_VisualFacet42.csd` | vision, 35-D FACET 4.2 |
| `CMU_MOSEI_COVAREP.csd` | audio, 74-D COVAREP |
| `CMU_MOSEI_Labels.csd` | sentiment labels |

Expected pickles next to this folder:

- `MOSEI/mosei_raw_bert.pkl` — text width 768
- `MOSEI/mosei_raw_glove.pkl` — text width 300

## MOSI computational sequences

Put these under `MOSI/cmumosi/`:

| File | Modality |
| --- | --- |
| `CMU_MOSI_TimestampedWords.csd` | text timings |
| `CMU_MOSI_Visual_Facet_42.csd` | vision, 35-D FACET 4.2 |
| `CMU_MOSI_COVAREP.csd` | audio, 74-D COVAREP |
| `CMU_MOSI_Opinion_Labels.csd` | opinion / sentiment labels |

Expected pickles:

- `MOSI/mosi_raw_bert.pkl`
- `MOSI/mosi_raw_glove.pkl`

## Loader reminder

`get_dataloader.py` drops rows whose text sums to 0, replaces `-inf` audio
with 0, optionally slices from the first non-zero text frame (`aligned=True`),
and pads to length **50** when `max_pad=True`. Ablations keep three slots and
zero the unused ones.

A synthetic pickle with the same keys: `python -m examples.inspect_pickle_schema`.
