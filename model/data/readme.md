# Feature files (not in git)

Personal checklist for the aligned MOSI / MOSEI dumps this repo expects.
Longer notes: [`docs/datasets.md`](../../docs/datasets.md).

## GloVe (rebuild only)

- `glove.840B.300d.txt` — Common Crawl 840B, 300-d. Only needed if you
  regenerate `*_raw_glove.pkl`. Runtime training reads the pickle, not
  this file.

## CMU-MOSEI raw computational sequences

Place under `MOSEI/cmumosei/`:

- `CMU_MOSEI_COVAREP.csd` — acoustic (74-d after alignment)
- `CMU_MOSEI_Labels.csd` — sentiment + emotion dimensions
- `CMU_MOSEI_TimestampedWords.csd` — word intervals used as the align-to
- `CMU_MOSEI_VisualFacet42.csd` — FACET 4.2 (35-d)

Training pickles the notebooks write:

- `MOSEI/mosei_raw_bert.pkl`
- `MOSEI/mosei_raw_glove.pkl`

## CMU-MOSI raw computational sequences

Place under `MOSI/cmumosi/`:

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

Training / transfer pickles:

- `MOSI/mosi_raw_bert.pkl`
- `MOSI/mosi_raw_glove.pkl`

## Pickle contract

Each pickle is a dict `train` / `valid` / `test` →
`vision`, `audio`, `text`, `labels`, `id`. Sequences are word-aligned and
typically already length 50. See `get_dataloader.py`.

Do not commit pickles, `.csd`, `.hdf5`, or GloVe vectors. `.gitignore`
already excludes them.
