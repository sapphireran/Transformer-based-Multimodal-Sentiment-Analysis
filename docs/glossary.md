# Glossary

Short lexicon for the MOSI / MOSEI notes in this repo.

**Acc-2 / Acc-5 / Acc-7**
: Classification accuracy after turning the continuous score into 2, 5,
or 7 labels. Acc-2 is the sign of the score (zeros dropped). Acc-5/7
use equal-width bins on `[-3, 3]`.

**Aligned clip**
: Visual, audio, and text already resampled onto the same time grid
(word-aligned via the CMU SDK). `Affectdataset` still trims leading
zeros on the text stream when `aligned=True`.

**BERT pickle**
: `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl`. Text width 768.

**COVAREP**
: Acoustic feature set (74-D after the preprocessing used here).

**Early fusion**
: Combine raw (or lightly projected) feature streams *before* a shared
sequence model. `ConcatEarly`, `TransformerEarly`.

**Facet 4.2**
: Visual facial-action features, 35-D in these pickles (`vision` key).

**GloVe pickle**
: `mosei_raw_glove.pkl` / `mosi_raw_glove.pkl`. Text width 300
(`glove.840B.300d`).

**GMTM**
: `GatedMultiTransfomerModel` — cross-modal transformer grid, learned
modality weights, sigmoid gates, attention pooling, linear head.

**Late fusion**
: Encode each modality on its own, then combine the clip vectors.
`ConcatLate`, `TransformerLate`, TFN, LRTF.

**LRTF**
: Low-Rank Tensor Fusion. Rank-constrained outer product.

**max_pad**
: Loader flag. `True` → every clip is exactly `max_seq_len` frames
(GMTM / TransformerEarly). `False` → dynamic pad + lengths (packed
LSTM/GRU).

**MOSEI**
: CMU Multimodal Opinion Sentiment and Emotion Intensity. Larger
YouTube-clip corpus; main training set in this repo.

**MOSI**
: CMU Multimodal Opinion-level Sentiment Intensity. Smaller movie-review
clip corpus; used here as a transfer test.

**MultiFramework**
: Thin wrapper: `encoders` (one per modality) + `fusion` + `head`.

**Non-zero protocol**
: Drop gold labels that are exactly 0 before binary F1 / Acc-2.

**Packed batch**
: `has_padding=True`. Encoders receive `(padded_tensor, lengths)` and
call `pack_padded_sequence`.

**TFN**
: Tensor Fusion Network. Full outer product with a homogeneous 1.

**Uniform bins**
: Acc-5/7 edges are equally spaced on `[-3, 3]`, not the official MOSI
histogram bins that isolate zero.
