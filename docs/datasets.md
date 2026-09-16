# Datasets and features

The experiments use the standard CMU multimodal sentiment corpora:

- **CMU-MOSI** — YouTube movie-review clips, utterance-level sentiment.
- **CMU-MOSEI** — larger in-the-wild collection, same label style.

Labels are continuous sentiment scores. The evaluation code treats them as
regression targets on roughly `[-3, 3]` and also buckets them for Acc7 / Acc5
/ Acc2. See [metrics.md](metrics.md).

This repo never checks the raw videos in. Everything the training scripts
see is a pre-aligned pickle.

## Pickle schema

`model/data/analysis_dataset.ipynb` prints the structure of
`mosei_raw_bert.pkl`:

```
{
  "train" | "valid" | "test": {
    "vision": ndarray [N, T, 35],
    "audio":  ndarray [N, T, 74],
    "text":   ndarray [N, T, 768 or 300],
    "labels": ndarray [N, 1, K],   # sentiment is labels[:, 0, 0]
    "id":     list[str]            # e.g. "-3g5yACwYnA[0]"
  }
}
```

Observed in that notebook:

- `data['valid']['text'][0].shape == (50, 768)` for the BERT pickle.
- Train split of MOSEI BERT has on the order of 16k utterance ids.
- Labels are sliced as `labels[:, 0, 0]` for histograms, which matches
  `_process_2` taking the first column when `label.shape[1] > 1`.

A synthetic pickle with the same keys and ranks can be built by
[`examples/tiny_affect_dataset.py`](../examples/tiny_affect_dataset.py).

## Feature sources

| Modality | MOSI field | MOSEI field | Width | Notes |
| --- | --- | --- | --- | --- |
| Visual | `CMU_MOSI_Visual_Facet_42` | `CMU_MOSEI_VisualFacet42` | 35 | Facial action features |
| Audio | `CMU_MOSI_COVAREP` | `CMU_MOSEI_COVAREP` | 74 | Acoustic LLDs |
| Text (words) | `CMU_MOSI_TimestampedWords` | `CMU_MOSEI_TimestampedWords` | — | alignment spine |
| Text (BERT) | encoded offline | encoded offline | 768 | `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl` |
| Text (GloVe) | `glove.840B.300d.txt` | same | 300 | `*_raw_glove.pkl` |
| Labels | `CMU_MOSI_Opinion_Labels` | `CMU_MOSEI_Labels` | 1 usable score | extra annotation channels may be present |

`model/data/MOSEI/get_mosei.py` is a CMU-MultimodalSDK recipe that aligns
everything to the word sequence (average-pool features onto word intervals),
then aligns to the label sequence, then writes HDF5. The checked-in path
hard-codes a Windows directory (`F:\MOSEI\...`) from the original personal
machine; treat it as a lab note, not a portable CLI.

## What `get_dataloader.py` does

### Drop empty text

`drop_entry` removes any utterance whose text tensor sums to 0. That protects
the aligned-start logic, which looks up `text.nonzero()[0][0]`.

### Alignment trim

`Affectdataset` with `aligned=True` (the default) finds the first nonzero
text row and slices vision / audio / text from that index. Unaligned mode
trims each modality independently.

### Padding modes

| Flag | Collate | Batch tensors | Used by |
| --- | --- | --- | --- |
| `max_pad=False` | `_process_1` | padded sequences + lengths + index + label | Concat / TFN / LMF / TransformerLate |
| `max_pad=True` | `_process_2` | stacked `[B, 50, F]` per modality + label | TransformerEarly, all GMTM scripts |

`max_pad` also truncates each sample to `max_pad_num` (default 50) and
right-pads with zeros.

### Optional z-normalization

`z_norm=True` standardizes each utterance along time, per feature, replacing
NaNs. The launch scripts leave this off. Audio `-inf` values are zeroed in
`Affectdataset.__init__`.

### Modality order in a batch

`_process_2` returns:

```
vision, audio, text, labels   # labels reshaped to [B, 1]
```

`MultiFramework` then feeds `inputs[0]` to encoder 0 (visual), and so on.
GMTM's `n_features` lists are therefore `[35, 74, 768]` or `[35, 74, 300]`,
**visual / audio / text**, even though the ablation *name* lists are written
`['text', 'audio', 'visual']`.

## Ablation masking

`get_ablation_dataloader` and `get_ablation_mosi_dataloader` do not rebuild
the model for a subset of modalities. They keep three slots and replace
dropped streams with zeros:

| Embedding | visual | audio | text |
| --- | --- | --- | --- |
| BERT | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| GloVe | `(50, 35)` | `(50, 74)` | `(50, 300)` |

So a `['text']` run still constructs `GatedMultiTransfomerModel(3, [35, 74, D])`.
The visual and audio projections see all zeros; the 3×3 attention grid still
runs. That is a stronger statement than "train a text-only network" — it is
"GMTM with two silent modalities".

## MOSI transfer loaders

`get_mosi_dataloader` and `get_ablation_mosi_dataloader` **concatenate
train + valid + test** into a single evaluation set:

```python
merged_test_data[key] = np.concatenate([
    alldata['train'][key],
    alldata['valid'][key],
    alldata['test'][key]
], axis=0)
```

The `mosi_test/` scripts then score MOSEI-trained checkpoints on that merged
MOSI tensor. That is not the official MOSI test split. Numbers in
`mosi_*_results.csv` should be read as "behavior on all MOSI utterances",
not as a leaderboard test score. This is called out again in
[results.md](results.md) and [personal_lab_notes.md](personal_lab_notes.md).

## Building a tiny compatible pickle

`examples/tiny_affect_dataset.py` writes a few synthetic utterances with:

- nonzero first text frames (so alignment does not crash)
- `labels` of shape `[N, 1, 1]`
- the same three feature widths as BERT or GloVe

Use it when you want to exercise collate logic without downloading CMU
features.
