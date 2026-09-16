# Architecture notes

Everything trainable lives in `model/models.py`. Training wraps three
pieces — **encoders**, **fusion**, **head** — inside `MultiFramework`
(`model/train_and_test.py`). GMTM is the exception: it *is* the fusion
block and already ends in a regression head, so the scripts pass
`Identity` for both encoder and head.

## MultiFramework

```
vision  ──► encoder_v ──┐
audio   ──► encoder_a ──┼──► fusion ──► head ──► ŷ ∈ ℝ
text    ──► encoder_t ──┘
```

`forward` accepts a list of modality tensors. If `has_padding=True`
(packed LSTM/GRU baselines), each encoder gets `[sequence, lengths]`.
GMTM and the early transformer use **fixed-length** padded tensors
(`max_pad=True` in the dataloader) and `has_padding=False`.

## Unimodal encoders

| Class | Role | Typical use |
| --- | --- | --- |
| `Identity` | Pass-through | GMTM, ConcatEarly, TransformerEarly |
| `LSTM` | Last hidden state, optional linear | ConcatLate |
| `GRU` / `GRUWithLinear` | Sequence or last step + projection | TensorFusion, LRTF |
| `Transformer` | 1×1 conv → 4-layer encoder → last step | leftover helper |
| `TransformerSeq` | Same stack, returns full `[B, T, D]` | TransformerLate unimodal towers |
| `MLP` | Two-layer ReLU MLP | heads |
| `AttentionPooling` | Softmax weights over time | GMTM temporal collapse |

`LSTM` / `GRUWithLinear` still contain the older `has_padding` path that
uses `pack_padded_sequence`. That is why `train_main_*.py` sets
`is_packed=True` for every fusion except `TransformerEarly`.

## Baseline fusion modules

### ConcatEarly

`torch.cat(modalities, dim=2)` on `[B, T, F_i]`. On BERT MOSEI this is
`35 + 74 + 768 = 877` channels. A packed LSTM+MLP then reads the joint
sequence.

### ConcatLate

Flatten each encoded vector and `cat` on `dim=1`. On BERT MOSEI the
ConcatLate LSTMs emit `64 + 256 + 1024 = 1344` units into an MLP head.

### TensorFusion

Implements the TFN outer-product trick from
[Zadeh et al.](https://github.com/Justin1904/TensorFusionNetworks):
prepend a `1` to each vector, then iteratively
`einsum('...i,...j->...ij')` and flatten. Three modalities of size
`(19, 39, 159)` produce a `20 * 40 * 160 = 128000`-D tensor — which is
why the TensorFusion head is `MLP(128000, 2048, 1)`.

### LowRankTensorFusion

Low-rank factorization of the same outer product
([Liu et al.](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)).
Each modality `x_i` is lifted with a leading 1, multiplied by a factor
`[rank, F_i+1, out_dim]`, then the rank slices are multiplied across
modalities and mixed with `fusion_weights`. Rank is 32 in the main
scripts.

### EarlyFusionTransformer

1×1 `Conv1d` from the concatenated channel count (877 BERT / 409 GloVe)
down to `embed_dim=32`, then a 4-layer `TransformerEncoder`. The
implementation returns the **last time step** of the encoder, not a
pooled sequence. The `MLP(64, 64, 1)` head in `train_main_bert.py` does
not match that 32-D output — treat the recorded TransformerEarly numbers
as coming from the checkpoint that was actually saved, and re-check the
head dim if you retrain.

### LateFusionTransformer

Unimodal `TransformerSeq` towers (`64 / 128 / 1024` on BERT) are
concatenated on the feature axis (`1216` D), projected to 32 D, then
another 4-layer encoder. Output is again the last time step.

`TransformerEncoderLayer(..., batch_first=True)` is constructed, but
`forward` still feeds `[T, B, D]`. That mismatch is leftover from the
original notes; it has not been cleaned up. New examples use GMTM's
custom encoder instead, which is consistently `[T, B, D]`.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the model this repo is named after.

Default study hyperparameters (`HParams` in `train_GMTM_*.py`):

| Knob | Value | Meaning |
| --- | --- | --- |
| `n_modalities` | 3 | visual, audio, text (always, even in ablations) |
| `n_features` | `[35, 74, 768]` or `[35, 74, 300]` | raw input widths |
| `embed_dim` | 64 | shared cross-modal width |
| `num_heads` | 4 | must divide `embed_dim` |
| `layers` | 4 | depth of each pairwise encoder |
| `attn_dropout_modalities` | `[0, 0, 0.1]` | extra dropout on text keys |
| `embed_dropout` | 0.2 | after the per-modality projection |
| `out_dropout` | 0.1 | classification head |
| `output_dim` | 1 | regression |

### 1. Per-modality projection

Each stream `x_i ∈ ℝ^{B×T×F_i}` is permuted to `[T, B, F_i]`, flattened
across time, and sent through

```
Linear(F_i → embed_dim) → LayerNorm → Dropout → ReLU
```

so every modality lives in the same 64-D space before attention.

### 2. Pairwise cross-modal transformers

For every ordered pair `(i, j)` a `TransformerEncoder` is run as
**query = modality i, key/value = modality j**. That is the
MulT-style directed cross-attention
(`x_in`, `x_in_k`, `x_in_v` in `TransformerEncoder.forward`).

There are `3 × 3 = 9` such stacks (including self-attention when
`i == j`). `trans_mems` is constructed but **not used** in the current
`forward` (the call is commented out).

Each encoder adds **sinusoidal positional embeddings**
(`SinusoidalPositionalEmbedding`) after scaling the input by
`sqrt(embed_dim)`.

### 3. Learned modality weights

The three directed outputs into target `i` are stacked and reduced
with a softmax over `self.modal_weights` (a length-3 parameter,
**shared across targets**):

```
h_fused_i = Σ_j softmax(w)_j · h_{i←j}
```

### 4. Per-modality gate

```
g_i = σ(W_i h_fused_i)
h_i ← g_i ⊙ h_fused_i
```

This is a residual-style *soft mute*: a modality can shrink its own
contribution without changing the other streams' parameters.

### 5. Concatenate, pool, regress

The three gated sequences are concatenated on the feature axis
(`[B, T, 192]`), passed through `AttentionPooling` (a linear → softmax
over time), then a two-layer head:

```
LayerNorm → Linear(192 → 96) → ReLU → Dropout → Linear(96 → 1)
```

A leftover `self.alpha` residual scalar is registered but unused.

### Ablations without shrinking the graph

Zeroing a modality in the dataloader (see `filter_modalities_list` in
`get_dataloader.py`) keeps the 9 pairwise encoders in place. The
projection of an all-zero stream is not exactly zero after LayerNorm +
ReLU, so "audio-off" is *soft* masking, not architectural removal.

## Custom transformer stack

`TransformerEncoder` / `TransformerEncoderLayer` are a small
fairseq-style port:

- pre-norm (`normalize_before = True`)
- `MultiheadAttention` with optional separate K/V
- FFN width `4 * embed_dim`, ReLU
- residual dropout on both sublayers

`create_attention_mask` currently returns `None` even when
`attn_mask=True`. `buffered_future_mask` exists but is unused. Do not
treat GMTM as a causal language model; it is bidirectional over the
clip.

## Complexity (order-of-magnitude)

With `T=50`, `D=64`, `L=4`, `M=3`:

- 9 encoders × 4 layers of attention is the dominant cost
- attention per layer is `O(T² D)` on the pairwise tensors
- TensorFusion's 128k-D head is the memory outlier among baselines

`train_and_test.all_in_one_train` reports wall time, peak RSS
(`memory_profiler`), and parameter count when `track_complexity=True`.

## What the examples exercise

`examples/gmtm_forward.py` builds a **tiny** GMTM (`embed_dim=16`,
`layers=1`, `num_heads=2`) and prints shapes after projection, fusion,
and the head. `examples/fusion_shapes.py` instantiates the baseline
modules with toy widths so you can see the outer-product blow-up
without loading MOSI.
