# Model architecture

This repository compares seven fusion families on the same three-modality
input: **vision** (FACET 4.2, 35-d), **audio** (COVAREP, 74-d), and **text**
(BERT 768-d or GloVe 300-d). Every family is assembled as

```
encoders  →  fusion  →  head
```

and wrapped by `MultiFramework` in `model/train_and_test.py`. The training
scripts in `model/train_main_*.py` and `model/train_GMTM_*.py` only change
which encoder / fusion / head objects are constructed.

## Shared building blocks (`model/models.py`)

| Module | Role |
| --- | --- |
| `Identity` | Pass-through encoder used when fusion sees raw features |
| `LSTM` / `GRU` / `GRUWithLinear` | Recurrent unimodal encoders (late fusion) |
| `Transformer` / `TransformerSeq` | 1×1 conv projection + 4-layer Transformer encoder |
| `MLP` | Two-layer prediction head |
| `AttentionPooling` | Softmax-weighted sum over the time axis |
| `SinusoidalPositionalEmbedding` | Fixed positional encodings for GMTM |

`Transformer` returns only the last time step. `TransformerSeq` returns the
full `[batch, time, dim]` sequence so a later fusion module can still attend.

## Fusion zoo

### Early concatenation (`ConcatEarly`)

The three sequences are concatenated on the feature axis
(`[B, T, 35+74+text_dim]`). A single LSTM (or the early Transformer below)
then reads the stacked stream. This is the cheapest multimodal baseline and
forces a shared temporal encoder.

### Late concatenation (`ConcatLate`)

Each modality is encoded independently (LSTM or Transformer), flattened, and
concatenated. The head never sees cross-modal interactions except through the
final MLP.

### Tensor Fusion (`TensorFusion`)

Implements the outer-product fusion from
[Zadeh et al., Tensor Fusion Network](https://github.com/Justin1904/TensorFusionNetworks).
Each modality vector is augmented with a constant `1`, then successive
Kronecker products produce a rank-3 (or higher) interaction tensor that is
flattened into the head. The BERT variant uses reduced encoder widths
`(19, 39, 159)` so the product stays at 128000 dimensions.

### Low-rank Tensor Fusion (`LowRankTensorFusion`)

The low-rank factorization from
[Liu et al., LMF](https://github.com/Justin1904/Low-rank-Multimodal-Fusion).
Each modality is multiplied by a learned factor of shape
`[rank, input_dim+1, output_dim]`; the factors are combined with a Hadamard
product and a rank-wise weight vector. Rank is 32 in the published scripts.

### Early Fusion Transformer (`EarlyFusionTransformer`)

A 1×1 convolution projects the concatenated feature dim (877 for BERT, 409
for GloVe) to `embed_dim = 32`. A 4-layer, 4-head Transformer encoder then
consumes the sequence. The published head is an MLP on the last step (BERT)
or `Identity` (GloVe).

### Late Fusion Transformer (`LateFusionTransformer`)

Each modality is encoded by `TransformerSeq` to `(64, 128, 1024)` (BERT) or
`(64, 128, 512)` (GloVe). The sequences are concatenated on the feature axis
and a second Transformer (input dim 1216 / 1792 → 32) reads the joint stream.

### Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the model this repo adds on top of the
baselines. For `n` modalities it builds an `n × n` grid of
`TransformerEncoder` blocks:

1. **Unimodal projection.** `Linear → LayerNorm → Dropout → ReLU` maps each
   modality from its native width onto a shared `embed_dim` (64 in the
   training scripts).
2. **Pairwise cross-attention.** Block `(i, j)` uses modality `i` as query
   and modality `j` as key/value, including the self-attention diagonal.
3. **Learned modality weights.** The `n` attended streams for target `i` are
   stacked and reduced with `softmax(modal_weights)`.
4. **Per-modality gate.** A sigmoid linear layer scales the fused stream so
   a noisy modality can shrink its contribution.
5. **Temporal pool.** The gated streams are concatenated to
   `[B, T, n * embed_dim]` and collapsed with `AttentionPooling`.
6. **Head.** `LayerNorm → Linear → ReLU → Dropout → Linear` regresses a
   scalar sentiment in `[-3, 3]`.

Default hyperparameters used in `train_GMTM_bert.py` / `train_GMTM_glove.py`:

| Field | Value |
| --- | --- |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` (vision, audio, text) |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |

Ablation runs keep the three encoder slots but zero the dropped modality
inside `get_ablation_dataloader`, so the same GMTM graph is used for every
subset.

## Tensor shapes (BERT, padded `T = 50`)

```
vision  [B, 50, 35 ]
audio   [B, 50, 74 ]
text    [B, 50, 768]
        ↓ modal_proj
        [T, B, 64] each
        ↓ 3×3 cross-transformers + gates
        [B, 50, 192]
        ↓ attention pool
        [B, 192]
        ↓ classification head
        [B, 1]
```

GloVe swaps the text width `768 → 300`; the rest of the GMTM graph is
unchanged.

## Device notes

Training scripts construct modules with `.cuda()`. The modules themselves
are device-agnostic except `LowRankTensorFusion`, which currently allocates
its ones-vector on `cuda:0` when a GPU is visible. The CPU examples under
`examples/` stay on CPU and avoid that path unless a GPU is present.
