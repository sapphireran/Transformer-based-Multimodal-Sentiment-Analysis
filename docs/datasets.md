# Datasets and on-disk layout

The trainers never read raw video. They consume **aligned pickle dictionaries** produced
from CMU-MOSI / CMU-MOSEI computational sequences.

## Expected pickle schema

Each of `mosi_raw_bert.pkl`, `mosi_raw_glove.pkl`, `mosei_raw_bert.pkl`,
`mosei_raw_glove.pkl` is a Python `dict`:

```
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray, "id": list},
  "valid": { ... },
  "test":  { ... },
}
```

| Key | Shape | Notes |
| --- | --- | --- |
| `vision` | `(N, T, 35)` | FACET 4.2. `-inf` audio is zeroed in `Affectdataset`. |
| `audio` | `(N, T, 74)` | COVAREP. |
| `text` | `(N, T, 768)` or `(N, T, 300)` | BERT-base last layer or GloVe 840B. |
| `labels` | `(N, 1, L)` or `(N, 1, 1)` | First column is the sentiment score used at train time. |
| `id` | length `N` | Segment ids such as `-3g5yACwYnA[0]`. |

`T` in the processed files used here is **50**. `get_dataloader` can further clip
with `max_seq_len`.

`drop_entry()` removes rows whose text sum is identically zero before a split is wrapped
in `Affectdataset`.

## Official sources

These are public academic datasets. Download them yourself from the CMU Multimodal SDK
distribution; this repo does not vendor the `.csd` files.

**CMU-MOSI** (expected under `model/data/MOSI/cmumosi/`):

- `CMU_MOSI_COVAREP.csd`
- `CMU_MOSI_Opinion_Labels.csd`
- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_Visual_Facet_42.csd`

**CMU-MOSEI** (expected under `model/data/MOSEI/cmumosei/`):

- `CMU_MOSEI_COVAREP.csd`
- `CMU_MOSEI_Labels.csd`
- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualFacet42.csd`

**GloVe** (optional, only if you rebuild GloVe pickles):

- `model/data/glove.840B.300d.txt` — the tracked file is an empty placeholder.
  Download [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) and replace it.

## How this repo builds the pickles

Notebooks (not the trainers) do the SDK work:

| Notebook / script | Output |
| --- | --- |
| [`model/data/MOSI/get_dataset.ipynb`](../model/data/MOSI/get_dataset.ipynb) | `mosi_raw_bert.pkl`, `mosi_raw_glove.pkl` |
| [`model/data/MOSEI/get_dataset.ipynb`](../model/data/MOSEI/get_dataset.ipynb) | MOSEI equivalents |
| [`model/data/MOSEI/get_mosei.py`](../model/data/MOSEI/get_mosei.py) | Older MOSI HDF5 export (Windows paths; not used by trainers) |
| [`model/data/analysis_dataset.ipynb`](../model/data/analysis_dataset.ipynb) | Inspect pickle keys and label histograms |

The MOSI notebook:

1. Loads the four computational sequences through `mmsdk`.
2. Aligns vision/audio to words, then to opinion labels.
3. Encodes words with `bert-base-uncased` **or** `torchtext` GloVe vectors, padded to 50.
4. Writes the split dictionaries using `DATASET.standard_folds`.

Place the resulting pickles next to the notebooks:

```
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
```

Trainers refer to those relative paths from `model/` or `model/mosi_test/`.

## Split sizes seen in this repo

From `analysis_dataset.ipynb` on a BERT MOSEI pickle (`mosei_raw_bert_old.pkl`):

| Split | Segments (`id` length) | Text width |
| --- | --- | --- |
| train | 16327 | 768 |
| valid | 1871 | 768 |
| test | 4662 | 768 |

CMU-MOSI is much smaller (standard SDK folds are on the order of ~1.2k / ~0.2k / ~0.7k
utterances). Exact counts depend on `drop_entry` and which videos fail alignment.

## Loader variants

All live in [`model/data/get_dataloader.py`](../model/data/get_dataloader.py).

| Function | Splits returned | Extra behavior |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | Standard MOSEI / MOSI training |
| `get_ablation_dataloader` | train, valid, test | Zeroes dropped modalities |
| `get_mosi_dataloader` | **one merged loader** | Concatenates train+valid+test, then wraps as `test` |
| `get_ablation_mosi_dataloader` | **one merged loader** | Same merge + modality zeroing |

The MOSI transfer scripts use the merged loaders on purpose: a MOSEI-trained checkpoint
is scored on **every** MOSI segment, not only the official test fold. That is a
transfer / domain-shift measurement, not the standard MOSI test protocol.
If you train **on MOSI** (the next step in [`mosi.md`](mosi.md)), use `get_dataloader`
so validation stays held out.

### Ablation zeroing

Dropped modalities are replaced with zeros of a fixed shape, not removed from the
batch. GMTM always sees three inputs:

| Embedding | vision | audio | text |
| --- | --- | --- | --- |
| `bert` | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| `glove` | `(50, 35)` | `(50, 74)` | `(50, 300)` |

So a "text-only" run is GMTM with silent audio/vision channels, not a smaller network.

## Alignment and normalization

`Affectdataset`:

- If `aligned=True` (default), all three modalities start at the first non-zero text frame.
- If `z_norm=True`, each modality is standardized over time per sample (NaNs → 0).
  Trainers currently leave `z_norm=False`.
- Audio entries equal to `-inf` are set to `0` in `__init__`.

## Synthetic stand-in

[`examples/01_synthetic_mosi_dataset.py`](../examples/01_synthetic_mosi_dataset.py)
writes a MOSI-shaped pickle that `Affectdataset` and `get_dataloader` can load.
Use it for CPU smoke tests when the official files are absent.

## Related

- MOSI transfer protocol and next experiments: [`mosi.md`](mosi.md)
- Metric definitions: [`evaluation.md`](evaluation.md)
