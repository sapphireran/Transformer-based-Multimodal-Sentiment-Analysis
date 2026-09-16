# Data directory

Personal notes for **CMU-MOSI** and **CMU-MOSEI** files used by this repo.
Longer explanation: [`../../docs/datasets.md`](../../docs/datasets.md).

Do **not** commit `.csd`, `.pkl`, `.hdf5`, or GloVe text files. Only the
loader code, notebooks, and this readme belong in git.

## GloVe

| File | Where to put it |
| --- | --- |
| `glove.840B.300d.txt` | Common crawl 840B / 300-D vectors, used when rebuilding `*_glove.pkl` |

Download from the [GloVe project page](https://nlp.stanford.edu/projects/glove/).
The notebooks that build pickles decide the exact local path.

## MOSEI

Place computational sequences under `MOSEI/cmumosei/`:

| File | Role |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | acoustic (74-D after the pickle build) |
| `CMU_MOSEI_Labels.csd` | continuous sentiment labels |
| `CMU_MOSEI_TimestampedWords.csd` | word timings / tokens |
| `CMU_MOSEI_VisualFacet42.csd` | visual Facet 4.2 (35-D after the pickle build) |

Processed outputs expected by the training scripts (not committed):

| File | Text width |
| --- | --- |
| `MOSEI/mosei_raw_bert.pkl` | 768 |
| `MOSEI/mosei_raw_glove.pkl` | 300 |

Builders:

- `MOSEI/get_dataset.ipynb` — pickle construction
- `MOSEI/get_mosei.py` — MMSDK align / HDF5 sketch with **machine-local**
  paths; edit before running

## MOSI

Place computational sequences under `MOSI/cmumosi/`:

| File | Role |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | acoustic |
| `CMU_MOSI_Opinion_Labels.csd` | labels |
| `CMU_MOSI_TimestampedWords.csd` | words |
| `CMU_MOSI_Visual_Facet_42.csd` | visual |

Processed outputs:

| File | Text width |
| --- | --- |
| `MOSI/mosi_raw_bert.pkl` | 768 |
| `MOSI/mosi_raw_glove.pkl` | 300 |

Builder: `MOSI/get_dataset.ipynb`.

## Loaders

`get_dataloader.py` is the only runtime dependency for training:

| Function | Returns | Notes |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | official-style three-way split |
| `get_ablation_dataloader` | train, valid, test | unused modalities zeroed |
| `get_mosi_dataloader` | **one** loader | concatenates MOSI train+valid+test |
| `get_ablation_mosi_dataloader` | **one** loader | same merge + zeroing |

`max_pad=True` → `_process_2` (fixed `T=50`).
`max_pad=False` → `_process_1` (packed lengths).

## Synthetic stand-in

If you only want tensor shapes, skip this folder and run
[`../../examples/synthetic_data.py`](../../examples/synthetic_data.py).
