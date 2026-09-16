# Glossary

Short definitions for the terms used in the other personal notes.

**Acc2** — Binary accuracy after dropping gold zeros and thresholding
the rest at 0. See [evaluation.md](evaluation.md).

**Acc5 / Acc7** — Accuracy after cutting `[-3, 3]` into 5 or 7 equal
bins. Not the same as some papers' uneven bin edges.

**Aligned** — `Affectdataset` flag. When true, visual and audio are
sliced from the first non-zero **text** row so the three streams share
a start index.

**BERT pickle** — Pre-extracted 768-d word / segment embeddings stored
in `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl`. No tokenizer runs at
train time.

**CMU-SDK / mmsdk** — Toolkit that stores multimodal streams as
computational sequences (`.csd`) and aligns them onto a common
interval grid.

**Collate `_process_1`** — Variable-length batch: padded sequences plus
lengths. Used with `is_packed=True`.

**Collate `_process_2`** — Fixed `(B, 50, F)` batch. Used with GMTM and
TransformerEarly.

**ConcatEarly** — Fuse by concatenating feature axes *before* a
sequence model.

**ConcatLate** — One unimodal encoder each, concatenate the clip
vectors, then an MLP.

**COVAREP** — Acoustic feature set (74-d in these pickles): pitch,
voicing, MFCCs, and related descriptors.

**Cross-modal attention** — In GMTM, queries from modality `i` attend
to keys/values from modality `j` for every pair `(i, j)`.

**Facet 4.2** — Visual action-unit / face descriptor stream (35-d
here).

**Gated fusion** — After mixing pair outputs, GMTM applies
`σ(W h) ⊙ h` per modality.

**GloVe 840B 300d** — Common Crawl word vectors. Text width becomes
300 instead of 768.

**GMTM** — `GatedMultiTransfomerModel` (the typo is in the class name).
Pairwise transformers + modality weights + gate + attention pooling.

**Identity encoder** — Pass-through module so fusion sees raw
sequences.

**`is_packed`** — `MultiFramework` flag. When true, each encoder
receives `[tensor, lengths]` and CuDNN is disabled for that step.

**Label / sentiment** — Continuous clip score, roughly `[-3, 3]`, from
human annotations. Regression target.

**Low-rank tensor fusion (LRF)** — Factor the multimodal tensor into
per-modality rank-`r` slices so you do not materialize the full outer
product.

**MOSEI** — CMU Multimodal Opinion Sentiment and Emotion Intensity.
Larger, in-the-wild, the main train set in this repo.

**MOSI** — CMU Multimodal Opinion-level Sentiment Intensity. Smaller;
used here as a transfer / eval set.

**`max_pad`** — If true, every clip is trimmed/padded to 50 steps
inside `__getitem__`.

**Modality dropout (HParams)** — Declared on GMTM HParams but not
implemented in `forward`. Ablations zero a whole stream instead.

**MultiFramework** — Thin `nn.Module` that owns `encoders`, `fusion`,
and `head`.

**Neutral / zero label** — Gold score exactly 0. Excluded from Acc2 / F1.

**Packed sequence** — `torch.nn.utils.rnn.pack_padded_sequence` so
LSTMs / GRUs skip pad steps.

**Pearson Corr** — Linear correlation between `y` and `ŷ` on the test
split.

**Tensor fusion** — Outer product of `[1; z_m]` across modalities;
captures multiplicative interactions at the cost of a huge vector.

**TransformerEarly / Late** — Shared-encoder-on-concat vs
per-modality-encoder-then-second-transformer.

**Unimodal ablation** — Keep GMTM's three slots, fill unused ones with
zeros, train as usual.

**`z_norm`** — Per-clip, per-feature standardization. Off in every
recorded script.
