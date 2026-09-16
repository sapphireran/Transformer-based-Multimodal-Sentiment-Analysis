# Datasets and features

All experiments use the CMU multimodal sentiment corpora distributed through
the [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

## CMU-MOSI

Short monologue clips of YouTube movie reviews. Each clip is annotated with a
continuous sentiment score in `[-3, 3]`. This repo treats MOSI primarily as a
**transfer / held-out** set: models are trained on MOSEI, then
`get_mosi_dataloader` evaluates them on the concatenation of MOSI train +
valid + test (see `model/mosi_test/`).

## CMU-MOSEI

The larger in-the-wild successor to MOSI (YouTube monologues, same `[-3, 3]`
scale). Main tables in `model/results/` and `model/main_results.csv` are
MOSEI test-set numbers.

## Feature views

Two text embeddings are cached as pickles. Acoustic and visual features are
shared across both views.

| View | Vision | Audio | Text | Typical pickle |
| --- | --- | --- | --- | --- |
| BERT | FACET 4.2, 35-d | COVAREP, 74-d | BERT, 768-d | `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl` |
| GloVe | FACET 4.2, 35-d | COVAREP, 74-d | GloVe 840B, 300-d | `mosei_raw_glove.pkl` / `mosi_raw_glove.pkl` |

Sequence length is truncated / padded to **50** word-aligned steps
(`max_seq_len=50`, `max_pad=True` for Transformer-early and GMTM).

## Pickle schema

`get_dataloader.py` expects a Python pickle of:

```python
{
    "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray},
    "valid": {...},
    "test":  {...},
}
```

`drop_entry()` removes clips whose text tensor is all zeros. `Affectdataset`
then:

1. trims leading padded frames so the first non-zero text (or per-modality
   non-zero, when `aligned=False`) is at index 0
2. optionally z-normalizes each clip independently
3. replaces `-inf` audio values with `0`
4. pads to `max_pad_num` when `max_pad=True`

Collate functions:

* `_process_1` — variable-length batch + lengths (used by recurrent late
  fusion; `is_packed=True`)
* `_process_2` — stacked `[B, T, F]` tensors (used by GMTM and early
  Transformer; `max_pad=True`)

## Ablation loaders

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` keep the three
slots but replace unselected modalities with zeros of the correct shape.
That lets GMTM stay a 3-input model while measuring text-only, audio+visual,
and so on.

## Rebuilding from the CMU SDK

Raw `.csd` computational sequences are **not** in git. The notebooks under
`model/data/MOSI/` and `model/data/MOSEI/` show the alignment recipe:

1. load TimestampedWords + FACET + COVAREP
2. align non-text modalities onto words with mean pooling
3. add opinion / sentiment labels and align again
4. dump an HDF5 or the pickle the dataloaders consume

See `model/data/readme.md` for the exact filenames.

## Synthetic substitute

`examples/toy_data.py` builds in-memory batches with the same shapes and
label range so the model graph and the metric helpers can be exercised
without downloading MOSI / MOSEI.
