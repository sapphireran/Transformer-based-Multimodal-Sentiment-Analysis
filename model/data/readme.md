# Feature files (not in git)

The training scripts read **pickle dumps** produced from the CMU Multimodal
SDK. Those dumps and the raw `.csd` sequences are local artifacts — only the
builder notebooks live in this folder. See [`docs/data-pipeline.md`](../../docs/data-pipeline.md)
for the in-memory contract the loaders expect.

## GloVe

### `glove.840B.300d.txt` storage location

Place the Common Crawl 840B / 300-d vectors next to the builder notebook (or
update the path inside `MOSI/get_dataset.ipynb` / `MOSEI/get_dataset.ipynb`)
if you want to rebuild `*_raw_glove.pkl`. The recorded GloVe experiments
consume the pickle, not this text file, at train time.

## MOSEI

Expected computational sequences under `MOSEI/cmumosei/`:

- `CMU_MOSEI_COVAREP.csd` — acoustic (74-d after the notebook)
- `CMU_MOSEI_Labels.csd` — sentiment / emotion labels
- `CMU_MOSEI_TimestampedWords.csd` — word stream used as the alignment spine
- `CMU_MOSEI_VisualFacet42.csd` — FACET 4.2 visual (35-d)

Builder: `MOSEI/get_dataset.ipynb`. SDK sketch: `MOSEI/get_mosei.py` (rewrite
the hard-coded `F:\...` paths first). Output pickles the scripts look for:

```text
MOSEI/mosei_raw_bert.pkl
MOSEI/mosei_raw_glove.pkl
```

## MOSI

Expected computational sequences under `MOSI/cmumosi/`:

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

Builder: `MOSI/get_dataset.ipynb`. Output pickles:

```text
MOSI/mosi_raw_bert.pkl
MOSI/mosi_raw_glove.pkl
```

`get_mosi_dataloader` concatenates MOSI train+valid+test before scoring. That
is intentional in the current transfer scripts and is **not** the official
MOSI test split.

## Peeking at a dump

`analysis_dataset.ipynb` prints keys and a few rows. Typical BERT train clip:

```text
vision [N, 50, 35]
audio  [N, 50, 74]
text   [N, 50, 768]
labels [N, 1, K]   # first value is the [-3, 3] sentiment score
```

CPU stand-in (no pickle required): `python3 examples/dataset_collate_demo.py`.
