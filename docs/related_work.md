# Where these fusion families come from

Personal notes for reading this repo next to the papers it borrows from.
The implementations live in `model/models.py`; they are research rewrites,
not vendor code.

## Tensor Fusion Network (TFN)

Zadeh et al., *Tensor Fusion Network for Multimodal Sentiment Analysis*
([code](https://github.com/Justin1904/TensorFusionNetworks)). Each unimodal
vector is augmented with a constant `1`, then successive outer products
build a tensor whose entries are all multiplicative interactions, including
the unimodal and bimodal slices (the extra `1` is what keeps those).

`TensorFusion` here is that product. The BERT script shrinks the encoder
outputs to `(19, 39, 159)` so the flattened tensor is 128000-d; GloVe uses
`(19, 39, 79)` → 64000-d. The cost is the MLP on top of that vector, not
the fusion module itself (the fusion has no parameters).

## Low-rank Multimodal Fusion (LMF)

Liu et al., *Efficient Low-rank Multimodal Fusion with Modality-Specific
Factors* ([code](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)).
Instead of materializing the full tensor, each modality is multiplied by a
learned factor of shape `[rank, dim+1, output_dim]`; the factors are
combined with a Hadamard product and a rank-wise weight.

`LowRankTensorFusion` uses rank 32. On the GloVe MOSEI table LMF is the
strongest **baseline** on Acc-2 / F1; GMTM still wins MAE / Acc-7 / Corr.

## Early vs late concatenation

The oldest multimodal baselines: stack features before a shared encoder
(`ConcatEarly`) or encode each stream and stack the embeddings
(`ConcatLate`). They are here as sanity checks. On MOSEI BERT they trail
the tensor and Transformer families by a clear MAE gap.

## Transformer early / late

Not MulT. **Early** concatenates raw features, projects with a 1×1 conv,
and runs a 4-layer Transformer over time. **Late** encodes each modality
with `TransformerSeq`, concatenates on the feature axis, and runs a second
Transformer (`LateFusionTransformer`). Late Transformer is the best
non-GMTM row on MOSEI BERT and on MOSI BERT transfer.

## Multimodal Transformer (MulT), for contrast

Tsai et al., *Multimodal Transformer* — pairwise **cross-modal attention**
(query from one stream, key/value from another) with a cyclic design and
self-attention memories. GMTM in this repo is in the same family
(pairwise `TransformerEncoder` blocks with `x_in`, `x_in_k`, `x_in_v`)
but then:

1. mixes the `n` attended streams with **learned softmax modality weights**
2. applies a **sigmoid gate** per target modality
3. **attention-pools** time instead of taking the last step
4. keeps dropped modalities as **zero tensors** so ablation does not
   change the graph

MulT is not imported; GMTM is the local variant those four choices define.

## Gated Multi-Transformer (GMTM)

The model this personal repo adds. See [architecture.md](architecture.md)
for the graph and [results.md](results.md) for the last numbers. The
useful empirical claim on the committed CSVs is narrow:

* on **MOSEI BERT**, full GMTM beats every listed baseline on every
  reported column
* **text** carries almost all of that gain; audio/visual alone are weak
* on **MOSI BERT transfer** (merged MOSI splits), Transformer-late is
  ahead of GMTM — so the MOSEI win does not automatically transfer
