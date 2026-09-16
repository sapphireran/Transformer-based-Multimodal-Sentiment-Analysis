# Architecture

This repository compares classical multimodal fusion blocks against a
gated cross-modal transformer, `GatedMultiTransfomerModel` (GMTM), on
utterance-level sentiment regression.

All modules live in [`model/models.py`](../model/models.py). The training
wrapper that stacks *encoder → fusion → head* is `MultiFramework` in
[`model/train_and_test.py`](../model/train_and_test.py).

## Tensor conventions

The pickle loaders emit one sample as `[vision, audio, text, label]`.

| Stream | Rank after padding | Feature width (BERT) | Feature width (GloVe) |
| --- | --- | --- | --- |
| vision (Facet 42) | `[B, T, 35]` | 35 | 35 |
| audio (COVAREP) | `[B, T, 74]` | 74 | 74 |
| text | `[B, T, D_text]` | 768 | 300 |
| label | `[B, 1]` | scalar in roughly `[-3, 3]` | same |

`T` is clipped and padded to 50 when `max_pad=True`. Packed-sequence
experiments keep variable lengths and pass `has_padding=True` into LSTM / GRU
encoders.

GMTM's `forward` expects a Python list `x` of length `n_modalities` where
`x[i]` is `[B, T, F_i]`. Internally it permutes each stream to `[T, B, F]`
before the transformer stack, then permutes back to `[B, T, embed_dim]`.

## MultiFramework

```
inputs[i]  →  encoders[i]  →  outs[i]
outs       →  fusion       →  fused
fused      →  head         →  scalar sentiment
```

- `encoders` is an `nn.ModuleList`. `Identity` means "fusion sees raw features".
- `fusion` can be a stateless concat, a tensor product, or GMTM.
- `head` is usually an MLP (or `Identity` when GMTM already ends in a linear
  classification head).
- If `has_padding` is true, each encoder receives `[tensor, lengths]`.

This is why the same three feature widths appear in every launch script:
the fusion block is written against a fixed modality order
`[visual, audio, text]`, even when an ablation zeros some of them out.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the model the ablation scripts train.

### Hyperparameters used in the launch scripts

```
num_heads = 4
layers = 4
embed_dim = 64
attn_dropout = 0.1
attn_dropout_modalities = [0, 0, 0.1]
relu_dropout = res_dropout = out_dropout = 0.1
embed_dropout = 0.2
attn_mask = True
output_dim = 1
```

`DefaultHyperParams` on the class itself uses a much smaller `embed_dim=9`
and `num_heads=3`. The experiment scripts always pass the `HParams` object
above.

### Forward pass, step by step

1. **Per-modality projection.** Each `x[i]` is flattened over time×batch,
   sent through `Linear(F_i → embed_dim) → LayerNorm → Dropout → ReLU`, then
   reshaped back to `[T, B, embed_dim]`. After this step every modality lives
   in the same width, so the cross-attention grids are square.
2. **Pairwise cross-modal transformers.** `self.trans[i][j]` is a
   `TransformerEncoder` that treats modality `i` as the query stream and
   modality `j` as key/value. For three modalities this is a 3×3 grid
   (self-attention on the diagonal, cross-attention off-diagonal).
3. **Learned modality weights.** The three outputs for target `i` are stacked
   and contracted with `softmax(self.modal_weights)`. The weights are shared
   across targets, not per-target.
4. **Gate.** `sigmoid(Linear(embed_dim → embed_dim))` multiplies the fused
   stream elementwise. This is the "gated" part of the name.
5. **Concatenation over modalities.** The three gated streams, now `[B, T, 64]`,
   are concatenated on the feature axis to `[B, T, 192]`.
6. **Attention pooling.** `AttentionPooling` scores each time step with a
   linear `192 → 1`, softmaxes over `T`, and returns `[B, 192]`.
7. **Classification head.** `LayerNorm → Linear(192 → 96) → ReLU → Dropout →
   Linear(96 → 1)`.

There is a leftover `self.alpha` parameter and commented residual path at the
bottom of `forward`. They are not used in the current return value.

`self.trans_mems` (self-attention "memory" stacks) is constructed but the
call is commented out:

```
#h_list = self.trans_mems[i](h_list)
```

Ablations do **not** shrink the 3-way grid. Unused modalities are replaced
with zeros of the original shape, so GMTM always sees three streams.

### Cross-attention implementation notes

`TransformerEncoder` adds sinusoidal positional embeddings, optional key/value
streams, and a stack of `TransformerEncoderLayer`. When `x_in_k` and `x_in_v`
are provided, the layer's `MultiheadAttention` uses those as key/value and the
projected source modality as query. That is the cross-modal path.

`attn_mask=True` is set in `HParams`, but `create_attention_mask` currently
returns `None`, so no causal / padding mask is applied.

## Classical fusion modules

These are the six methods in `train_main_bert.py` / `train_main_glove.py`.

### ConcatEarly

`torch.cat(modalities, dim=2)` on `[B, T, F_i]` → `[B, T, Σ F_i]`.

BERT widths: `35 + 74 + 768 = 877`. GloVe: `35 + 74 + 300 = 409`.

The main BERT script then runs `LSTM(877, 1024, has_padding=True)` + `MLP`.
`TransformerEarly` is a different early-fusion path (see below).

### ConcatLate

Each modality is encoded first (LSTM in the main sweep), flattened, then
`torch.cat(..., dim=1)`. BERT late widths: `64 + 256 + 1024 = 1344`.

### TensorFusion

Implements the Zadeh et al. Tensor Fusion Network outer product, including a
leading 1 so unimodal and bimodal terms appear in the product. For three
vectors of size `a, b, c` the result has width `(a+1)(b+1)(c+1)`. The BERT
script therefore shrinks the GRU outputs to `19, 39, 159` so the product is
`20 × 40 × 160 = 128000` before the MLP.

### LowRankTensorFusion

Low-rank factorization of the same tensor product (Liu et al.). Each modality
`m` has a factor of shape `[rank, F_m + 1, output_dim]`. The forward pass
appends a 1, multiplies by the factor, and takes a product over modalities,
then a rank-wise weighted sum. BERT config: inputs `[32, 64, 256]`, output
256, rank 32.

### EarlyFusionTransformer

Concatenates on the feature axis (same as ConcatEarly), projects with
`Conv1d(Σ F → 32, kernel=1)`, runs a 4-layer `nn.TransformerEncoder`
(`nhead=4`), and returns the last time step. The class sets
`batch_first=True` on the encoder layer but then feeds `[T, B, 32]`. The
main BERT head is `MLP(64, 64, 1)` after this block; the GloVe script uses
`Identity` as the head.

### LateFusionTransformer

Expects already-encoded sequences concatenated on the last axis, projects
with `Conv1d(in_dim → 32)`, runs the same 4-layer encoder, and returns the
last step `[B, 32]`. BERT `in_dim=1216` (`64+128+1024` from per-modality
`TransformerSeq`). GloVe `in_dim=704` in the architecture math
(`64+128+512`) but the launch script passes `1792` — see
[personal_lab_notes.md](personal_lab_notes.md).

### TransformerSeq / Transformer

`TransformerSeq` is the per-modality encoder used before late transformer
fusion: `Conv1d(F → dim)` then a 4-layer encoder, returning the full
sequence `[B, T, dim]`. `Transformer` is the same stack but returns only
the last time step.

## Other building blocks

| Module | Role |
| --- | --- |
| `LSTM` / `GRU` / `GRUWithLinear` | Sequence encoders; optional `pack_padded_sequence` |
| `MLP` | Two-layer head, optional dropout |
| `AttentionPooling` | Time-weighted sum used by GMTM |
| `TransformerFusion` | Alternate fusion that stacks modality vectors as a short sequence |
| `SinusoidalPositionalEmbedding` | GMTM positional encodings |
| `Identity` | Pass-through encoder or head |

`models.py` defines both a `class Linear` and a later `def Linear(...)`
that overwrites the name. The function (Xavier-uniform init) is what
`TransformerEncoderLayer` actually calls.

## Shape walkthrough (BERT, padded, GMTM)

```
vision  [B, 50, 35 ] ─┐
audio   [B, 50, 74 ] ─┼─→ modal_proj ─→ [50, B, 64] each
text    [B, 50, 768] ─┘
                         ─→ 3×3 TransformerEncoder
                         ─→ softmax weights + gate
                         ─→ concat [B, 50, 192]
                         ─→ attention pool [B, 192]
                         ─→ head [B, 1]
```

A CPU walkthrough that prints every intermediate rank is
[`examples/fusion_shape_walkthrough.py`](../examples/fusion_shape_walkthrough.py).
