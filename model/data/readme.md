# Data layout

Personal notes for the MOSI / MOSEI files this project expects. The
`.csd` / `.pkl` / GloVe objects themselves are **not** in git. See
[`docs/datasets.md`](../../docs/datasets.md) for the full write-up and
`model/synthetic.py` if you only want matching tensor ranks.

## GloVe

`glove.840B.300d.txt` — 300-D Common Crawl vectors. Used when a script
passes `embedding='glove'` or loads `mosei_raw_glove.pkl` /
`mosi_raw_glove.pkl`.

## MOSEI (`MOSEI/cmumosei/`)

| File | Stream |
| --- | --- |
| `CMU_MOSEI_COVAREP.csd` | acoustic, 74-D after alignment |
| `CMU_MOSEI_VisualFacet42.csd` | Facet 4.2 visual, 35-D |
| `CMU_MOSEI_TimestampedWords.csd` | word timings (alignment target) |
| `CMU_MOSEI_Labels.csd` | continuous sentiment labels |

Derived caches the training scripts load:

- `MOSEI/mosei_raw_bert.pkl` — text width 768
- `MOSEI/mosei_raw_glove.pkl` — text width 300

`MOSEI/get_mosei.py` is a CMU Multimodal SDK sketch (paths still point
at a local `F:\MOSEI\...` machine). Update `MOSI_PATH` / output paths
before running it.

## MOSI (`MOSI/cmumosi/`)

| File | Stream |
| --- | --- |
| `CMU_MOSI_COVAREP.csd` | acoustic |
| `CMU_MOSI_Visual_Facet_42.csd` | Facet 4.2 |
| `CMU_MOSI_TimestampedWords.csd` | words |
| `CMU_MOSI_Opinion_Labels.csd` | opinion labels |

Derived caches:

- `MOSI/mosi_raw_bert.pkl`
- `MOSI/mosi_raw_glove.pkl`

`mosi_test/` evaluates MOSEI-trained checkpoints on these pickles.
`get_mosi_dataloader` concatenates train+valid+test into one test set.

## Loader entry points (`get_dataloader.py`)

- `get_dataloader` — train / valid / test
- `get_ablation_dataloader` — same, unused modalities zero-filled
- `get_mosi_dataloader` — merged MOSI test set
- `get_ablation_mosi_dataloader` — merged + zero-fill

`max_pad=True` stacks clips to `max_seq_len` (50). `max_pad=False`
returns packed lengths for LSTM/GRU encoders.
