# Data layout (personal MOSI / MOSEI checkout)

This folder is where the original training scripts look for **aligned pickles** and, if you rebuild features, CMU-MultimodalSDK `.csd` files. None of those binaries are in git.

A longer write-up is in [`docs/datasets.md`](../../docs/datasets.md). Synthetic stand-ins used by `examples/` do not live here.

## Expected pickles

| File | Used by |
| --- | --- |
| `MOSEI/mosei_raw_bert.pkl` | `train_main_bert.py`, `train_GMTM_bert.py` |
| `MOSEI/mosei_raw_glove.pkl` | `train_main_glove.py`, `train_GMTM_glove.py` |
| `MOSI/mosi_raw_bert.pkl` | `mosi_test/train_mosi_bert.py`, `mosi_test/mult_bert_mosi.py` |
| `MOSI/mosi_raw_glove.pkl` | `mosi_test/train_mosi_glove.py`, `mosi_test/mult_glove_mosi.py` |

Each pickle is a dict with `train` / `valid` / `test` splits. A split has `vision`, `audio`, `text`, `labels`, and usually `id`. After `drop_entry`, text-all-zero clips are gone.

Feature widths the ablation loader hard-codes:

| embedding | visual | audio | text | time |
| --- | --- | --- | --- | --- |
| `bert` | 35 | 74 | 768 | 50 |
| `glove` | 35 | 74 | 300 | 50 |

## GloVe

`glove.840B.300d.txt` in this directory is an **empty placeholder**. Rebuild language features with the real [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) file (about 2 GB).

## SDK computational sequences

### MOSEI (`MOSEI/cmumosei/`)

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

### MOSI (`MOSI/cmumosi/`)

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

`MOSEI/get_mosei.py` and the two `get_dataset.ipynb` notebooks align words → FACET / COVAREP (mean pool) → labels. Paths inside those files still point at the original Windows machine; edit them before rerunning.

## Dataloader entry points (`get_dataloader.py`)

| Function | Returns | Notes |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | Standard MOSEI / MOSI split |
| `get_ablation_dataloader` | train, valid, test | Zero-masks missing modalities |
| `get_mosi_dataloader` | one loader | Concatenates all MOSI splits (transfer test) |
| `get_ablation_mosi_dataloader` | one loader | Transfer test + zero-mask |

`max_pad=True` stacks fixed `[50, F]` clips (`_process_2`). `max_pad=False` uses `pad_sequence` and returns lengths (`_process_1`).
