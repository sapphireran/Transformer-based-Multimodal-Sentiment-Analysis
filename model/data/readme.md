# Feature files

This folder holds **local** CMU-MOSI / CMU-MOSEI dumps. The large `.csd`,
`.pkl`, and GloVe text files are gitignored. Training scripts only import
`get_dataloader.py`.

A full description of the pickle schema, alignment, and ablation zero-out
is in [`docs/datasets.md`](../../docs/datasets.md).

## Expected pickle names (after you build or copy them)

```
MOSEI/mosei_raw_bert.pkl     # text width 768
MOSEI/mosei_raw_glove.pkl    # text width 300
MOSI/mosi_raw_bert.pkl
MOSI/mosi_raw_glove.pkl
```

Each pickle is a dict `{'train','valid','test'}` →
`{'vision','audio','text','labels'}` (optional `id`). Widths are 35 / 74 /
768-or-300. Sequences are cut or padded to 50 steps in the Dataset.

## GloVe

`glove.840B.300d.txt` — word vectors used only if you rebuild text from
`TimestampedWords` instead of using a pre-baked BERT pickle. Put the file
anywhere and point the notebook at it.

## MOSEI raw MMSDK sequences (`MOSEI/cmumosei/`)

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

Rebuild notebook: `MOSEI/get_dataset.ipynb`. Scratch alignment script:
`MOSEI/get_mosei.py` (contains machine-local Windows paths).

## MOSI raw MMSDK sequences (`MOSI/cmumosi/`)

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

Rebuild notebook: `MOSI/get_dataset.ipynb`.

## Peeking at a pickle

`analysis_dataset.ipynb` prints split sizes and a few rows. The BERT-MOSEI
train split that notebook saw had 16327 utterances.
