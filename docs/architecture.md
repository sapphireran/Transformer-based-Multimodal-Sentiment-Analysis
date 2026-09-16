# Architecture

All neural modules live in [`model/models.py`](../model/models.py). Training
wraps them in `MultiFramework` (`encoders → fusion → head`) from
[`model/train_and_test.py`](../model/train_and_test.py).

GMTM is a special case: the fusion module already contains projections, the
cross-modal stack, temporal pooling, and the regression head. The corresponding
train scripts therefore pass `Identity` encoders and an `Identity` head.

```text
                    ┌────────────┐
 visual [B,T,35]  → │ encoder_v  │─┐
 audio  [B,T,74]  → │ encoder_a  │─┼→ fusion → head → ŷ [B,1]
 text   [B,T,D]   → │ encoder_t  │─┘
                    └────────────┘
 D = 768 (BERT) or 300 (GloVe)
 T ≤ 50 after padding / clipping
```

## Feature widths used everywhere

| Stream | Source | Width |
| --- | --- | --- |
| Visual | FACET 4.2 | 35 |
| Audio | COVAREP | 74 |
| Text (BERT) | `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl` | 768 |
| Text (GloVe) | `mosei_raw_glove.pkl` / `mosi_raw_glove.pkl` | 300 |

Early-fusion concatenated width is `35+74+768 = 877` (BERT) or
`35+74+300 = 409` (GloVe).

## Encoders

| Module | Input | Output | Notes |
| --- | --- | --- | --- |
| `Identity` | anything | same | Used when fusion sees raw sequences |
| `LSTM` | `[B,T,F]` or packed `(x, lengths)` | last hidden, flattened | `has_padding=True` packs with `pack_padded_sequence` |
| `GRU` / `GRUWithLinear` | same | sequence or last hidden, then optional linear | TFN / LMF encoders |
| `MLP` | `[B,F]` | `[B, out]` | Late heads; optional per-layer dump |
| `Transformer` | `[B,T,F]` | `[B, embed]` | 1×1 conv to `embed`, 4-layer encoder, **last time step only** |
| `TransformerSeq` | `[B,T,F]` | `[B,T, embed]` | Same stack, keeps the time axis |
| `EarlyFusionTransformer` | `[B,T,F_cat]` or list | `[T last, B, 32]` indexed as `[-1]` → `[B,32]` | `embed_dim = 32`, 4 heads, 4 layers |
| `LateFusionTransformer` | `[B,T,F_cat]` or list | `[B, 32]` | Projects concatenated encoder outputs |

`Transformer` / `TransformerSeq` feed the encoder in **sequence-first** layout
`[T,B,E]`, which matches the default `nn.TransformerEncoderLayer`.
`EarlyFusionTransformer` and `LateFusionTransformer` construct the layer with
`batch_first=True` but still permute to `[T,B,E]` before the encoder. Treat
that as an implementation quirk: the module runs, but the batch and time axes
are swapped relative to what `batch_first=True` advertises. The examples print
the actual output shape so you can see what the head receives.

## Fusion modules

### `ConcatEarly`

`torch.cat(modalities, dim=2)` — streams must already share `[B, T, *]`.
Used with `Identity` encoders and an LSTM+MLP head on the 877-d / 409-d
concatenated sequence.

### `ConcatLate`

Flattens each stream from `dim=1` and concatenates on the feature axis.
Used after per-modality LSTMs. BERT hidden sizes `64 + 256 + 1024 = 1344`;
GloVe `64 + 256 + 512 = 832`.

### `TensorFusion` (TFN)

Appends a constant-1 to each vector and takes the successive outer product.
Three modalities with (19, 39, 159) features become
`20 × 40 × 160 = 128000` (BERT) or `20 × 40 × 80 = 64000` (GloVe). The head is
a wide MLP that reads that flattened tensor. Memory-heavy; that is why LMF
exists.

### `LowRankTensorFusion` (LMF)

Low-rank factors `(rank, in_dim+1, out_dim)` per modality, multiplied in the
rank dimension, then mixed by `fusion_weights`. Rank is 32 in both BERT and
GloVe sweeps. Output dim is 256 (BERT) or 128 (GloVe).

### `TransformerFusion`

Stacks `[B, F]` modality vectors as a short sequence of length `n_modalities`,
runs a small encoder, mean-pools, projects. Not used in the recorded CSV
sweeps; included as a spare block.

## GMTM — `GatedMultiTransfomerModel`

Class name keeps the original spelling (`Transfomer`). Recorded hyperparameters
(from `train_GMTM_*.py`):

| Field | Value |
| --- | --- |
| `n_modalities` | 3 |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `attn_dropout` | 0.1 (memory / self path) |
| `attn_dropout_modalities` | `[0, 0, 0.1]` |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |

### Forward pass

Inputs `x[i]` are `[B, T, F_i]`.

1. **Project.** Each stream goes through `Linear(F_i, 64) → LayerNorm → Dropout → ReLU`.
   Internally the time and batch axes are flattened (`[T*B, F]`) and restored to
   `[T, B, 64]`.
2. **Pairwise cross-modal transformers.** For every ordered pair `(i, j)` a
   `TransformerEncoder` runs with query = modality `i` and key/value = modality
   `j`. That is a full `3 × 3` grid, including self-attention on the diagonal.
   Each encoder is the custom `TransformerEncoder` in `models.py` (sinusoidal
   positions, pre-norm, residual FFN with width `4 * embed_dim`).
3. **Modality mix.** The three pair outputs for a fixed `i` are stacked and
   reduced with `softmax(modal_weights)` via `einsum`.
4. **Gate.** `h = σ(W_i h) ⊙ h` for each `i`.
5. **Concat.** `[B, T, 64]` streams become `[B, T, 192]`.
6. **Attention pooling.** A linear `192 → 1` produces time weights; the weighted
   sum is `[B, 192]`.
7. **Head.** `LayerNorm → Linear(192, 96) → ReLU → Dropout → Linear(96, 1)`.

```text
x_v, x_a, x_t          [B, T, F_i]
        │
        ▼
   modal_proj[i]       [T, B, 64]
        │
        ▼
 ┌─────────────────────────────┐
 │ trans[i][j](q=i, k=j, v=j)  │   3×3 cross-modal grid
 └─────────────────────────────┘
        │
        ▼
 softmax(modal_weights) mix
        │
        ▼
 sigmoid gate
        │
        ▼
 concat over i           [B, T, 192]
        │
        ▼
 AttentionPooling        [B, 192]
        │
        ▼
 classification_head     [B, 1]
```

`trans_mems` (a second self-attention stack per modality) is constructed but
the call is commented out in `forward`. `alpha` is a leftover residual scalar
that is also unused. Ablation does **not** shrink `n_modalities`; dropped
streams are replaced with zeros of the official shape (see the data docs).

## `MultiFramework`

```python
outs = [encoders[i](inputs[i]) for i in range(n)]
fused = fusion(outs)
return head(fused)
```

When `has_padding=True` (`is_packed=True` in `train()`), each encoder receives
`[tensor, lengths]` and packed LSTM/GRU paths are used. Transformer-early
runs with `is_packed=False` because it wants a dense `[B, T, F]` stack.

## Baseline wiring (MOSEI BERT)

Exact constructor calls from `train_main_bert.py`:

| Fusion | Encoders | Fusion args | Head |
| --- | --- | --- | --- |
| ConcatEarly | 3× Identity | `ConcatEarly` | `LSTM(877,1024)` + `MLP(1024,1024,1)` |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | `ConcatLate` | `MLP(1344,1344,1)` |
| LMF | GRUWithLinear → 32 / 64 / 256 | `LowRankTensorFusion([32,64,256], 256, 32)` | `MLP(256,256,1)` |
| TFN | GRUWithLinear → 19 / 39 / 159 | `TensorFusion` | `MLP(128000,2048,1)` |
| TransformerEarly | 3× Identity | `EarlyFusionTransformer(n_features=877)` | `MLP(64,64,1)` |
| TransformerLate | TransformerSeq 35→64, 74→128, 768→1024 | `LateFusionTransformer(in_dim=1216)` | `MLP(32,32,1)` |

GloVe uses the same pattern with smaller text towers. See
[`experiments.md`](experiments.md) for the GloVe table and for the fact that
several scripts currently **evaluate** a saved `.pt` rather than train.

## Worked CPU example

[`examples/gmtm_forward.py`](../examples/gmtm_forward.py) builds a
`B=4, T=16` synthetic clip and prints the tensor at each of the seven steps
above. [`examples/fusion_shapes.py`](../examples/fusion_shapes.py) does the
same for every fusion module.
