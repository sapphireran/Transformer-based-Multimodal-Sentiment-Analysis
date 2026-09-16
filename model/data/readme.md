# Data layout (personal)

Raw CMU files and aligned pickles stay **off git**. This folder only holds loaders, notebooks, and notes. See [docs/datasets.md](../../docs/datasets.md) for shapes, alignment, and the loader API.

## GloVe

| File | Where |
| --- | --- |
| `glove.840B.300d.txt` | Common Crawl 840B / 300-d. Put it in `model/data/` (or update the notebook path). Not committed. |

Word lookup is used only when **rebuilding** `*_raw_glove.pkl`. Training itself just reads the pickle.

## MOSEI (`.csd` → pickle)

Put the CMU-SDK computational sequences in `MOSEI/cmumosei/`:

| File | Stream |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | acoustic, 74-d after alignment |
| `CMU_MOSEI_Labels.csd` | sentence scores in [-3, 3] |
| `CMU_MOSEI_TimestampedWords.csd` | word intervals |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2, 35-d after alignment |

Build `MOSEI/mosei_raw_bert.pkl` and `MOSEI/mosei_raw_glove.pkl` with `MOSEI/get_dataset.ipynb`. The pickle contract is:

```text
{train,valid,test} → {vision, audio, text, labels}
vision [N, T, 35]
audio  [N, T, 74]
text   [N, T, 768 or 300]
```

`get_mosei.py` in this folder is a MOSI alignment script with a hard-coded Windows path. Read it as a personal scratch file; prefer the notebooks or [docs/reproducing.md](../../docs/reproducing.md).

## MOSI (`.csd` → pickle)

Put sequences in `MOSI/cmumosi/`:

| File | Stream |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | acoustic |
| `CMU_MOSI_Opinion_Labels.csd` | clip scores in [-3, 3] |
| `CMU_MOSI_TimestampedWords.csd` | word intervals |
| `CMU_MOSI_Visual_Facet_42.csd` | FACET 4.2 |

Build `MOSI/mosi_raw_bert.pkl` and `MOSI/mosi_raw_glove.pkl` with `MOSI/get_dataset.ipynb`.

`get_mosi_dataloader` **merges train+valid+test** so a MOSEI checkpoint can be scored on every remaining MOSI clip. That is the transfer protocol used by `model/mosi_test/`, not the official MOSI test split.

## Loaders

`get_dataloader.py` is the only runtime dependency of the train scripts:

- `get_dataloader` — official splits (MOSEI training)
- `get_ablation_dataloader` — official splits, unused modalities zeroed
- `get_mosi_dataloader` — merged MOSI pool
- `get_ablation_mosi_dataloader` — merged MOSI pool, unused modalities zeroed

`drop_entry` drops clips whose text tensor is all zeros. Audio `-inf` is replaced with 0.

## What you can run without any of this

```bash
python examples/run_all_examples.py
```

Synthetic tensors in `examples/common.py` reuse the 35 / 74 / 768 (or 300) widths so fusion + GMTM can be smoke-tested on CPU.
