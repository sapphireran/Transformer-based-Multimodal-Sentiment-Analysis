## Local feature files

These paths are **not** in git. They are the files the training scripts
expect on the machine that actually trains. Layout notes and pickle
schema: [docs/data-pipeline.md](../../docs/data-pipeline.md).

## GLOVE

### `glove.840B.300d.txt` storage location

Put the Common Crawl 300-d vectors where the GloVe feature-build
notebook can see them. The training scripts themselves read
`MOSEI/mosei_raw_glove.pkl` / `MOSI/mosi_raw_glove.pkl`, not this file
directly.

## MOSEI

Expected pickle (after you build or copy it):

- `MOSEI/mosei_raw_bert.pkl` — vision 35, audio 74, text 768
- `MOSEI/mosei_raw_glove.pkl` — vision 35, audio 74, text 300

Raw CMU-SDK sequences in `MOSEI/cmumosei/`:

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

## MOSI

Expected pickle:

- `MOSI/mosi_raw_bert.pkl`
- `MOSI/mosi_raw_glove.pkl`

Raw sequences in `MOSI/cmumosi/`:

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

If you only want to inspect modules, skip this directory and run
`examples/` from the repo root.
