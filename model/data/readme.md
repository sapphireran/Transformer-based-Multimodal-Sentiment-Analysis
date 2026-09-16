# Feature files (personal workstation checklist)

The training scripts never read the `.csd` files directly. They expect
aligned pickles produced once on a local machine:

| Pickle (under this folder) | Text | Visual | Audio |
| --- | --- | --- | --- |
| `MOSEI/mosei_raw_bert.pkl` | 768-d BERT | 35-d Facet | 74-d COVAREP |
| `MOSEI/mosei_raw_glove.pkl` | 300-d GloVe | 35-d Facet | 74-d COVAREP |
| `MOSI/mosi_raw_bert.pkl` | 768-d BERT | 35-d Facet | 74-d COVAREP |
| `MOSI/mosi_raw_glove.pkl` | 300-d GloVe | 35-d Facet | 74-d COVAREP |

Those pickles are intentionally not in git. A tiny compatible stand-in
can be written with:

```bash
python examples/tiny_affect_dataset.py --embedding bert --write-pkl /tmp/tiny_mosei_bert.pkl
```

Schema, alignment, and the MOSI merge-all-splits behavior are documented
in [`docs/datasets.md`](../../docs/datasets.md).

## GloVe

`glove.840B.300d.txt` — Common Crawl 840B / 300-d. The tracked file in
this repo is an empty placeholder. Put the real vectors here only if you
rebuild text features from `TimestampedWords`.

## MOSEI computational sequences (`MOSEI/cmumosei/`)

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

`MOSEI/get_mosei.py` aligns words → labels via CMU-MultimodalSDK and
writes HDF5. Paths inside that script are from the original Windows
machine (`F:\MOSEI\...`) and need editing before a rerun.

## MOSI computational sequences (`MOSI/cmumosi/`)

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

Interactive SDK sessions: `MOSI/get_dataset.ipynb`, `MOSEI/get_dataset.ipynb`.
Pickle inspection and label histograms: `analysis_dataset.ipynb`.
