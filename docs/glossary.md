# Glossary

Short personal glossary for the terms used in the notes and examples.

**Acc-2.** Binary accuracy after dropping gold-zero clips and taking
`sign(score) > 0`. Same protocol as `eval_affect`.

**Acc-5 / Acc-7.** Classification accuracy after cutting [-3, 3] into 5 or
7 equal-width bins (`split_uniform_*`). Not the integer-label MOSI scheme
used in some papers.

**Aligned features.** Vision / audio frames collapsed onto word intervals
so all three tensors share a time index. Assumed throughout this repo.

**BERT pickle.** `text` width 768. Produced offline; this repo does not
call Hugging Face at train time.

**Collate `_process_1`.** Variable-length pad + lengths. Used with packed
RNNs (`is_packed=True`).

**Collate `_process_2`.** Fixed `T=50` stack. Used with transformers and
GMTM.

**ConcatEarly / ConcatLate.** Fuse by `torch.cat` either on time-aligned
features (`dim=2`) or on flattened encoder states (`dim=1`).

**COVAREP.** Acoustic feature set (74-d here) shipped with CMU-MOSI/MOSEI.

**Facet 4.2.** Visual facial-action features (35-d here).

**GloVe pickle.** `text` width 300, from `glove.840B.300d`.

**GMTM.** `GatedMultiTransfomerModel`: pairwise cross-modal transformers,
softmax modality weights, sigmoid gates, attention pooling.

**Homogenization.** Prefixing a 1 onto a modality vector so a tensor
product can represent all polynomial terms up to degree `n`. Used by
TensorFusion and LowRankTensorFusion.

**`is_packed`.** `MultiFramework` flag. When true, encoders receive
`[tensor, lengths]` and `train()` disables cuDNN on that forward.

**Low-rank tensor fusion (LRTF).** CP-style factorization of the
multimodal tensor product; rank 32 in the scripts.

**MAE.** Mean absolute error on the raw [-3, 3] score. Training loss.

**MOSEI.** CMU multimodal opinion-level sentiment dataset (large). Main
benchmark in this repo.

**MOSI.** Smaller predecessor. Used here as a transfer pool; loaders merge
splits unless you call `get_dataloader` yourself.

**`MultiFramework`.** Thin `encoders → fusion → head` module.

**Packed sequence.** `pack_padded_sequence` so an LSTM/GRU skips pad
steps. Requires `_process_1` lengths.

**Pearson Corr.** `scipy.stats.pearsonr` between gold and predicted
scalars.

**TensorFusion (TF).** Full outer-product fusion; head sees tens of
thousands of features.

**TransformerEarly / TransformerLate.** Shared-over-time transformer on
concatenated raw features vs. per-modality `TransformerSeq` plus a late
transformer.

**Word-aligned.** See aligned features.

**Zero ablation.** Unused modality replaced by `zeros(50, F)` so GMTM’s
`n=3` graph stays intact.
