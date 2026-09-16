# Datasets and dataloaders

The training scripts never read raw video. They read **aligned pickle dictionaries** produced from CMU-MOSI and CMU-MOSEI via [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Those pickles are not in git (they are hundreds of MB). `model/data/glove.840B.300d.txt` is an empty placeholder; you still need the real GloVe file to rebuild language features.

## Feature files the notebooks expect

Listed in `model/data/readme.md` and the SDK field names used in `model/data/MOSEI/get_mosei.py`.

### CMU-MOSEI (`model/data/MOSEI/cmumosei/`)

| File | Stream |
| --- | --- |
| `CMU_MOSEI_TimestampedWords.csd` | word-level text (alignment spine) |
| `CMU_MOSEI_VisualFacet42.csd` | FACET 4.2 visual (35-d after processing) |
| `CMU_MOSEI_COVAREP.csd` | COVAREP acoustic (74-d) |
| `CMU_MOSEI_Labels.csd` | sentiment / emotion labels |

### CMU-MOSI (`model/data/MOSI/cmumosi/`)

| File | Stream |
| --- | --- |
| `CMU_MOSI_TimestampedWords.csd` | words |
| `CMU_MOSI_Visual_Facet_42.csd` | FACET 4.2 |
| `CMU_MOSI_COVAREP.csd` | COVAREP |
| `CMU_MOSI_Opinion_Labels.csd` | opinion / sentiment labels |

`get_mosei.py` aligns non-text streams onto words with mean pooling, then aligns to the label sequence, and can dump HDF5. The notebooks `get_dataset.ipynb` (MOSI and MOSEI) are the interactive version of that pipeline. Paths inside those notebooks are local Windows paths from the original machine (`F:\MOSEI\...`); rewrite them before rerunning.

## Pickle contract

`get_dataloader()` does `pickle.load` and expects:

```
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray, "id": [...]},
  "valid": { ... },
  "test":  { ... },
}
```

`model/data/analysis_dataset.ipynb` inspected a BERT MOSEI pickle with:

| Split | Clips (after whatever filtering was applied in that file) |
| --- | --- |
| train | 16,327 |
| valid | 1,871 |
| test | 4,662 |

Text rows in that notebook were `(50, 768)`. Labels were stored as a wider annotation vector; the dataloader takes `labels[ind]` and, when the last dim is `> 1`, keeps the first column as the regression target (`_process_1` / `_process_2`).

## Cleaning

`drop_entry(dataset)` deletes any clip whose **text** tensor is all zeros (`k.sum() == 0`). It applies the same index drop to every key in that split.

`Affectdataset` also replaces `-inf` in audio with `0.0` (COVAREP often uses `-inf` for unvoiced frames).

Optional per-clip z-norm (`z_norm=True`) standardizes each modality along time. The recorded scripts leave this **off**.

## Alignment trim

If `aligned=True` (default), the dataset finds the first non-zero **text** row and slices vision / audio / text from that index. That drops leading pad that the pickle may have inserted before the first word.

## Two batch formats

### Variable length — `max_pad=False`

`_process_1` pads each modality independently with `pad_sequence` and also returns lengths and clip ids:

```
modalities, lengths, ids, labels
```

Used by concat / tensor / late-transformer entries (`is_packed=True` in `train()`).

### Fixed length — `max_pad=True`

`_process_2` stacks already-padded `[50, F]` tensors:

```
vision, audio, text, labels
```

Used by transformer-early and GMTM.

## MOSI test loaders

`get_mosi_dataloader` and `get_ablation_mosi_dataloader` concatenate train+valid+test into one evaluation set. That is intentional for the `mosi_test/` scripts: they measure how a **MOSEI-trained** checkpoint scores on all MOSI clips, not a MOSI-only test split.

## Modality ablation

`get_ablation_dataloader(..., modalities=['text', 'audio'], embedding='bert')` walks every sample and, for any stream **not** in `modalities`, writes a zero tensor of the official shape:

| embedding | visual | audio | text |
| --- | --- | --- | --- |
| `bert` | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| `glove` | `(50, 35)` | `(50, 74)` | `(50, 300)` |

The model graph stays 3-input. See [ablation.md](ablation.md).

## Synthetic stand-in

`examples/lib/synthetic.py` builds MOSI/MOSEI-shaped batches (same dims, random features, labels in `[-3, 3]`) so the docs examples can run in this checkout without SDK files.
