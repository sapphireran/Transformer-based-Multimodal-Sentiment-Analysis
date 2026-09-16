# Datasets and features

The training scripts never read raw video. They consume **aligned
multimodal pickles** produced from the CMU Multimodal SDK computational
sequences. This page is the map from those public files to the tensors
`Affectdataset` yields.

## MOSI vs MOSEI

| | CMU-MOSI | CMU-MOSEI |
| --- | --- | --- |
| Content | YouTube movie-review clips | YouTube monologues, many topics |
| Typical size | ~2.2k labeled segments | ~23k labeled segments |
| Label | opinion intensity, roughly `[-3, 3]` | sentiment intensity, `[-3, 3]` |
| Split keys in the pickle | `train` / `valid` / `test` | same |
| Text embedding files in this repo | `model/data/MOSI/mosi_raw_bert.pkl`, `mosi_raw_glove.pkl` | `model/data/MOSEI/mosei_raw_bert.pkl`, `mosei_raw_glove.pkl` |

Both corpora use the same three computational sequences:

| Modality | MOSI field | MOSEI field | Width after alignment |
| --- | --- | --- | ---: |
| Language (words) | `CMU_MOSI_TimestampedWords` | `CMU_MOSEI_TimestampedWords` | 768 (BERT) or 300 (GloVe) |
| Visual | `CMU_MOSI_Visual_Facet_42` | `CMU_MOSEI_VisualFacet42` | 35 |
| Acoustic | `CMU_MOSI_COVAREP` | `CMU_MOSEI_COVAREP` | 74 |
| Label | `CMU_MOSI_Opinion_Labels` | `CMU_MOSEI_Labels` | scalar |

Place the `.csd` files exactly where `model/data/readme.md` says:

```text
model/data/MOSEI/cmumosei/CMU_MOSEI_COVAREP.csd
model/data/MOSEI/cmumosei/CMU_MOSEI_Labels.csd
model/data/MOSEI/cmumosei/CMU_MOSEI_TimestampedWords.csd
model/data/MOSEI/cmumosei/CMU_MOSEI_VisualFacet42.csd

model/data/MOSI/cmumosi/CMU_MOSI_COVAREP.csd
model/data/MOSI/cmumosi/CMU_MOSI_Opinion_Labels.csd
model/data/MOSI/cmumosi/CMU_MOSI_TimestampedWords.csd
model/data/MOSI/cmumosi/CMU_MOSI_Visual_Facet_42.csd
```

GloVe vectors (`glove.840B.300d.txt`) go in `model/data/`. The file in
git is a **placeholder**; you need the real 2 GB dump from the Stanford
GloVe release to rebuild the GloVe pickles.

## Pickle schema

`get_dataloader` does `pickle.load` and expects:

```python
{
    "train": {"text": ndarray, "audio": ndarray, "vision": ndarray, "labels": ndarray},
    "valid": {...},
    "test":  {...},
}
```

Each modality array is `[N, T_raw, F]`. Labels are `[N, 1, 1]` or
`[N, 1]` in the files this repo was developed against; the collate
functions flatten them to `[B, 1]`.

`drop_entry` removes any clip whose **text** sums to 0 (empty
transcript). Audio `-inf` values (common in COVAREP) are replaced with
0 inside `Affectdataset`.

## Alignment and cropping

`Affectdataset.__getitem__` assumes the three streams are already
word-aligned (the CMU SDK `align` step). If `aligned=True` (the
default) it finds the first non-zero text timestep and crops vision /
audio / text from that index. That drops the silent pad that the SDK
leaves at the front of a segment.

Optional per-clip z-normalization (`z_norm=True`) standardizes each
feature independently over time and replaces NaNs with 0. The training
scripts leave this **off**.

## Two collate paths

`get_dataloader(..., max_pad=False)` uses `_process_1`:

- pads each modality independently to the longest clip in the **batch**
- also returns per-modality lengths and the original index
- LSTM / GRU encoders run with `has_padding=True` / `is_packed=True`

`get_dataloader(..., max_pad=True)` uses `_process_2`:

- every clip is cropped/padded to `max_seq_len` (default 50) **inside
  `__getitem__`**
- the batch is a stack: `vision, audio, text, labels`
- used by `TransformerEarly` and GMTM

`get_mosi_dataloader` is the transfer helper: it concatenates MOSI
train+valid+test into one test loader so a MOSEI-trained checkpoint can
be scored on every MOSI clip. That is why the MOSI CSV tables are not
a standard held-out MOSI split.

## Ablation loaders

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` take a list
such as `['text', 'audio']`. Dropped modalities are replaced with
**zeros of the official shape**, not removed. GMTM therefore always
sees three inputs; a unimodal run is "two modalities are the zero
tensor".

Zero shapes:

| Embedding | visual | audio | text |
| --- | --- | --- | --- |
| BERT | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| GloVe | `(50, 35)` | `(50, 74)` | `(50, 300)` |

That is why ablation training **requires** `max_pad=True`. A packed
loader cannot zero-fill a variable-length dropped stream as cleanly.

## Rebuilding pickles from `.csd`

`model/data/MOSEI/get_mosei.py` is a MOSI-shaped sketch (the field
names inside it are MOSI names) that:

1. Loads the word / FACET / COVAREP computational sequences through
   `mmsdk`.
2. Aligns visual and acoustic frames onto words with a simple average
   collapse.
3. Adds the label sequence and aligns to it (keeping time).
4. Writes an HDF5 dump.

The notebooks `get_dataset.ipynb` under `MOSI/` and `MOSEI/` are the
interactive version of the same recipe, including BERT / GloVe
embedding. They still contain machine-local paths (`F:\MOSEI\...`).
Copy the notebook, point `MOSI_PATH` / `MOSEI_PATH` at your `.csd`
directory, and write the pickle next to the training scripts'
expected path.

A minimal `mmsdk` recipe:

```python
from mmsdk import mmdatasdk as md

recipe = {
    "CMU_MOSEI_TimestampedWords": "cmumosei/CMU_MOSEI_TimestampedWords.csd",
    "CMU_MOSEI_VisualFacet42":    "cmumosei/CMU_MOSEI_VisualFacet42.csd",
    "CMU_MOSEI_COVAREP":          "cmumosei/CMU_MOSEI_COVAREP.csd",
}
dataset = md.mmdataset(recipe)
dataset.align("CMU_MOSEI_TimestampedWords", collapse_functions=[lambda i, f: f.mean(0)])
```

Then embed the aligned words with BERT or GloVe, attach labels, and
`pickle.dump` the `{train, valid, test}` dict.

Official CMU download pages:

- MOSI: http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- MOSEI: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
- SDK: https://github.com/CMU-MultiCOMP-Lab/MMCore

## What the synthetic examples use instead

`examples/synthetic_data.py` does **not** download anything. It draws
`N` clips of length `T=16` (or 50) from a standard normal, writes a
label in `[-3, 3]`, and optionally zeros a modality so you can exercise
the ablation path. Shapes match the real loaders, which is enough to
test fusion modules, GMTM, and the metric helpers.
