# Data directory (personal machine)

Git does not hold the real features. This folder is the *contract* for
paths the training scripts already hardcode.

Longer explanation of loaders, padding, and the MOSI merge:
[`../../docs/data-pipeline.md`](../../docs/data-pipeline.md).
Local install checklist: [`../../docs/setup.md`](../../docs/setup.md).

## GloVe

### `glove.840B.300d.txt`

Stanford GloVe 840B, 300-d. The copy in git is a **0-byte stub**. Drop
the real file here (or symlink it) before rebuilding `*_raw_glove.pkl`.
The main train scripts do not read this txt; they read the pickles.

## MOSEI (`MOSEI/`)

Expected pickle products (not in git):

```text
MOSEI/mosei_raw_bert.pkl
MOSEI/mosei_raw_glove.pkl
```

If I rebuild from the CMU SDK, the `.csd` files live in
`MOSEI/cmumosei/`:

| File | Stream |
| ---- | ------ |
| `CMU_MOSEI_COVAREP.csd` | audio, 74-d after my processing |
| `CMU_MOSEI_Labels.csd` | sentiment (and extras I ignore) |
| `CMU_MOSEI_TimestampedWords.csd` | alignment spine |
| `CMU_MOSEI_VisualFacet42.csd` | vision, 35-d in the pickle |

Notebooks: `MOSEI/get_dataset.ipynb`. `MOSEI/get_mosei.py` is an old
Windows scratch script that still says MOSI in the paths — do not run
it as-is.

`analysis_dataset.ipynb` last showed **16327** train ids on the BERT
pickle after `drop_entry`.

## MOSI (`MOSI/`)

Expected pickle products (not in git):

```text
MOSI/mosi_raw_bert.pkl
MOSI/mosi_raw_glove.pkl
```

SDK files in `MOSI/cmumosi/`:

| File | Stream |
| ---- | ------ |
| `CMU_MOSI_COVAREP.csd` | audio |
| `CMU_MOSI_Opinion_Labels.csd` | sentiment |
| `CMU_MOSI_TimestampedWords.csd` | alignment spine |
| `CMU_MOSI_Visual_Facet_42.csd` | vision |

Notebook: `MOSI/get_dataset.ipynb`.

`get_mosi_dataloader` **merges train+valid+test**. Transfer numbers in
`../mosi_test/` are not a held-out MOSI test. Details in
[`../../notes/04-mosi-transfer.md`](../../notes/04-mosi-transfer.md).

## Feature widths the rest of the repo assumes

| Stream | BERT pickle | GloVe pickle |
| ------ | ----------- | ------------ |
| visual | `(T, 35)` | `(T, 35)` |
| audio | `(T, 74)` | `(T, 74)` |
| text | `(T, 768)` | `(T, 300)` |
| T after max-pad | 50 | 50 |

Order in every batch: visual, audio, text, then label.

## What I do not keep here

Raw mp4 / wav, HuggingFace caches, extra SDK highlevel fields I never
wired (`OpenFace`, `glove_vectors` from the SDK itself, etc.).
