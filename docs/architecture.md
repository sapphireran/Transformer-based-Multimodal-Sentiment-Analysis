# Architecture

This note describes the modules in `model/models.py` and how
`MultiFramework` in `model/train_and_test.py` wires them together. It is
written against the code as it exists in this repo, not against a paper
abstraction.

## High-level graph

Every trainable experiment is the same three-stage graph:

```text
vision [B, T, 35] ──► encoder_v ──┐
audio  [B, T, 74] ──► encoder_a ──┼──► fusion ──► head ──► ŷ [B, 1]
text   [B, T, D_t] ──► encoder_t ──┘
```

`D_t` is `768` for BERT and `300` for GloVe. `T` is variable when the
loader uses packed sequences (`_process_1`) and is fixed at 50 when the
loader uses max-padding (`_process_2`).

`MultiFramework.forward` does three things:

1. Run each encoder. If `has_padding` is true, the encoder receives
   `[tensor, lengths]` so LSTM / GRU can call `pack_padded_sequence`.
2. Store the encoder outputs on `self.reps` and pass them to `fusion`.
3. Store the fused tensor on `self.fuseout` and pass it to `head`.

That extra bookkeeping exists so a custom objective can regularize
intermediate representations. The default objective is
`torch.nn.L1Loss` on the scalar prediction.

## Encoders

### Identity

Pass-through. Used whenever fusion itself is expected to consume the raw
sequence (early concat, early Transformer, GMTM).

### LSTM / GRU / GRUWithLinear

Sequence encoders that collapse time to a vector (or keep the last hidden
state). Important constructor flags:

| Flag | Meaning |
| --- | --- |
| `has_padding` | Pack the sequence with the length tensor from the collate fn |
| `dropout` / `dropoutp` | Dropout after the recurrent core |
| `flatten` | Flatten leftover axes before the optional linear map |
| `linear_layer_outdim` (LSTM) / `outdim` (GRUWithLinear) | Project the hidden state |

`GRUWithLinear` is the encoder used by tensor-fusion variants because
those methods need a **small** per-modality vector. TensorFusion in
particular explodes as `∏ (d_i + 1)`, so the BERT setup projects to
`(19, 39, 159)` before the outer product (`20 * 40 * 160 = 128000`).

### Transformer and TransformerSeq

Both apply a bias-free `Conv1d` (`kernel_size=1`) to map `n_features → embed_dim`, then a 4-layer, 4-head `nn.TransformerEncoder`.

- `Transformer` returns **only the last timestep** (`x[-1]` after the
  encoder). It is a sequence-to-vector encoder.
- `TransformerSeq` returns the **full sequence** `[B, T, embed_dim]`.
  Late fusion needs this because `LateFusionTransformer` concatenates
  those sequences on the feature axis.

Input convention: `[B, T, F]`. The conv layer permutes to `[B, F, T]`.

### EarlyFusionTransformer / LateFusionTransformer

These are fusion modules that happen to be Transformers. They are
documented in [fusion-methods.md](fusion-methods.md). One implementation
detail is worth repeating here: both classes construct
`TransformerEncoderLayer(..., batch_first=True)` but then feed the
encoder a `[T, B, D]` tensor (the result of `permute(2, 0, 1)`). The
layer still runs; the `batch_first` flag is simply inconsistent with the
actual layout. When you change these modules, pick one convention and
use it on both the layer and the permute.

## GMTM: Gated Multi-Transformer

Class: `GatedMultiTransfomerModel`.

### Hyperparameters

`DefaultHyperParams` inside the class is **not** what the training
scripts use. The live values live on the `HParams` object in
`train_GMTM_bert.py` / `train_GMTM_glove.py`:

| Field | Default in class | Value used in scripts |
| --- | ---: | ---: |
| `num_heads` | 3 | 4 |
| `layers` | 3 | 4 |
| `embed_dim` | 9 | 64 |
| `attn_dropout` | 0.1 | 0.1 |
| `attn_dropout_modalities` | `[0] * 1000` | `[0, 0, 0.1]` |
| `relu_dropout` / `res_dropout` | 0.1 | 0.1 |
| `out_dropout` | 0.0 | 0.1 |
| `embed_dropout` | 0.25 | 0.2 |
| `output_dim` | 1 | 1 |
| `attn_mask` | True | True |

`modality_dropout` and `use_text_transformer` are declared on `HParams`
but **not read** by `GatedMultiTransfomerModel`. They are leftovers.

### Per-modality projection

```text
Linear(n_features[i], embed_dim) → LayerNorm → Dropout → ReLU
```

The forward pass permutes each input from `[B, T, F]` to `[T, B, F]`,
flattens `(T * B, F)` through the projection, then reshapes back to
`[T, B, embed_dim]`. After this step every modality lives in the same
width, which is what makes the pairwise grid legal.

### Pairwise cross-modal grid

`self.trans` is an `n × n` module list. Cell `(i, j)` is a
`TransformerEncoder` that treats modality `i` as the query stream and
modality `j` as key/value:

```text
h_ij = trans[i][j](proj_x[i], proj_x[j], proj_x[j])
```

The custom `TransformerEncoder` adds sinusoidal positional embeddings,
optional pre-norm, multi-head attention, and a 4×-wide ReLU FFN. When
`x_in_k` / `x_in_v` are supplied it becomes **cross-attention**; when
they are omitted it is self-attention.

`self.trans_mems` (a per-modality self-attention stack) is constructed
but the call that would use it is commented out:

```text
# h_list = self.trans_mems[i](h_list)
```

So the memory Transformers do **not** affect the current numbers.

### Softmax fusion + gate

For each target modality `i` the `n` attended streams are stacked and
reduced with a learned softmax weight vector `modal_weights` of length
`n_modalities`. A sigmoid gate `σ(W_i h)` then scales the fused stream
elementwise. This is the "gated" part of GMTM: a modality can down-weight
its own fused representation if the cross-modal evidence is weak.

### Temporal pooling and head

The `n` gated streams are concatenated on the feature axis
(`[B, T, n * embed_dim]`) and collapsed over time by
`AttentionPooling`: a linear `d → 1` score, softmax over `T`, and a
weighted sum. The classification head is

```text
LayerNorm → Linear(d, d/2) → ReLU → Dropout → Linear(d/2, 1)
```

A residual path (`self.alpha`, plus a commented linear map of the pooled
vector) is **not** applied. `self.alpha` is a leftover parameter.

### Complexity

With 3 modalities, 4 layers, and 4 heads, GMTM builds `3 × 3 = 9`
cross-modal encoders plus 3 unused memory encoders. That is the main
reason it is heavier than `TransformerLate`, which encodes each modality
once and fuses afterwards.

## Custom Transformer building blocks

These sit under GMTM and are not used by the `nn.TransformerEncoder`
baselines:

| Symbol | Role |
| --- | --- |
| `SinusoidalPositionalEmbedding` | Classic 10000-base sin/cos, padding index 0 |
| `TransformerEncoderLayer` | Pre-norm MHA + residual FFN |
| `buffered_future_mask` | Upper-triangular `-inf` mask (defined, not currently wired) |
| `CustomLinear` / `CustomLayerNorm` | Xavier-uniform linear and a thin LN wrapper |

`make_positions` is the helper that turns a padded index tensor into
positions for the sinusoidal table.

There is a name collision in `models.py`: a `class Linear` is defined
early, then later a function `def Linear(...)` overwrites that name in
the module dict. GMTM's encoder layers call the **function**, which
returns an initialized `nn.Linear`. Do not instantiate `models.Linear`
expecting the class.

## AttentionPooling

Independent of GMTM, this is a reusable time-pooling block:

```text
scores = softmax(Linear(F → 1)(x), dim=time)     # [B, T]
out    = sum(x * scores.unsqueeze(-1), dim=time) # [B, F]
```

It is what lets GMTM consume a variable-length (or padded) sequence
without taking only the last hidden state.

## Device and packing conventions

- Training scripts call `.cuda()` on every module. The train loop
  itself falls back to CPU via
  `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`.
- `LowRankTensorFusion.forward` builds its ones-vector on
  `cuda:0` if any CUDA device is visible, otherwise CPU. Keep the
  module and the inputs on the same device.
- Packed vs. padded is **not** a model property. It is a property of
  the dataloader (`max_pad`) plus the `is_packed` flag on `train()` /
  `test()`. `TransformerEarly` is the only baseline that uses
  `max_pad=True` and `is_packed=False`. GMTM also uses `max_pad=True`
  because it wants a dense `[B, T, F]` cube.

## What is *not* in the graph

- No modality dropout at the input (the `HParams.modality_dropout`
  field is unused).
- No word-level language Transformer on top of BERT/GloVe; text arrives
  already embedded.
- No speaker / video-id embeddings.
- No multi-task heads (emotion, intensity, etc.). The only target is
  the scalar MOSI/MOSEI sentiment score.
