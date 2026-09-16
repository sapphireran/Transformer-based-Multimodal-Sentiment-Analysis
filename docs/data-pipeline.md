# Data pipeline

How utterances get from CMU `.csd` files to a batch the model can see.

## On-disk products

I do not train from `.csd` in the main scripts. The notebooks collapse the SDK
output into pickles:

```text
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray, "id": list},
  "valid": { ... },
  "test":  { ... }
}
```

Shapes I last inspected in `analysis_dataset.ipynb` (MOSEI BERT):

- `text[i]` is `(50, 768)` after truncation / pad
- `vision[i]` is `(50, 35)`
- `audio[i]` is `(50, 74)`
- MOSEI train had **16327** ids in that notebook run

GloVe pickles swap text to `(T, 300)`.

`drop_entry` removes utterances whose text tensor sums to 0 (empty text). Audio
`-inf` is replaced with 0.0 inside `Affectdataset`.

Optional per-utterance z-norm exists (`z_norm=True`) but the train scripts leave it
off.

## Two batching modes

### Packed (`max_pad=False`) → `_process_1`

Used by Concat / TFN / LMF / TransformerLate.

Each sample keeps its true length. The collate function `pad_sequence`s each
modality and also returns lengths. `MultiFramework` with `has_padding=True` packs
those lengths into the LSTM/GRU.

Sample tuple: `[vision, audio, text, index, label]`.

### Max-pad (`max_pad=True`) → `_process_2`

Used by TransformerEarly and GMTM.

Each modality is truncated to `max_pad_num=50` and zero-padded to exactly 50.
Collate **stacks** to a dense `(B, 50, F)`. No index is returned.

Sample tuple: `[vision, audio, text, label]`.

TransformerEarly needs this because `EarlyFusionTransformer` concats on dim 2 and
does not know how to pack. GMTM’s pairwise attention is written for dense `T × B × D`.

## Alignment trim

If `aligned=True` (default), `__getitem__` finds the first nonzero text row and
slices **all three** modalities from that index. The idea is to drop leading pad
that the pickle still carries. If text is all zeros, the code prints and `exit()`s —
those rows should already have been removed by `drop_entry`.

## Ablation zero-fill

`get_ablation_dataloader` / `get_ablation_mosi_dataloader` do **not** drop encoder
slots. They walk each sample and replace unselected modalities with

```text
torch.zeros(50, F)     # F = 35 / 74 / 768 or 300
```

GMTM still builds a 3-way pairwise grid. Audio-only therefore means “audio is real,
text and vision are exact zeros, but every cross-attn block still runs.”

That is why I do not read an audio-only row as “a unimodal audio model.” It is
“GMTM with two modalities clamped to zero.” The gates *can* learn to ignore the
zeros; they do not have to.

## MOSI transfer loader (this one is easy to misread)

`get_mosi_dataloader` and `get_ablation_mosi_dataloader`:

1. Load the MOSI pickle.
2. `drop_entry` on train / valid / test.
3. **Concatenate train + valid + test** into one `merged_test_data`.
4. Return a **single** DataLoader over that merge.

So every MOSI CSV in `model/mosi_test/` is scored on the **entire MOSI corpus**,
including utterances the original MOSI split would have called train. I did this
when I wanted a quick “does a MOSEI checkpoint even point the right direction on
MOSI clips” check. It is **not** a paper-style MOSI test number.

If I ever train on MOSI properly, I must stop using these two functions as-is.

## Label tensor shapes

MOSEI labels in the pickle are wider than a scalar (the notebook dump shows a
vector including things that look like emotion / intensity extras). `_process_1`
and `_process_2` take `sample[-1].reshape(...)[0]` when `shape[1] > 1`, i.e. they
keep the **first** row as the regression target. That first entry is the sentiment
score I trained on. I should still re-verify against the SDK field names before
quoting a paper comparison.

## Feature recipes I used when rebuilding from the SDK

From `model/data/readme.md` and `get_mosei.py`:

**MOSEI**

- `CMU_MOSEI_VisualFacet42`
- `CMU_MOSEI_COVAREP`
- `CMU_MOSEI_TimestampedWords`
- `CMU_MOSEI_Labels`

**MOSI**

- `CMU_MOSI_Visual_Facet_42`
- `CMU_MOSI_COVAREP`
- `CMU_MOSI_TimestampedWords`
- `CMU_MOSI_Opinion_Labels`

Alignment is word-level (`dataset.align(text_field, collapse_functions=[avg])`),
then aligned again to the label segments. Visual 35-d is the processed Facet
subset I ended up with in the pickle, not the raw 42-d name.

## GloVe file

`model/data/glove.840B.300d.txt` is a stub in git. The real Stanford file is ~5 GB.
The notebooks that build `*_raw_glove.pkl` are the only consumers.
