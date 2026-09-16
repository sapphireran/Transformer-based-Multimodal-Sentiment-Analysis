# Architecture

This note describes the modules in [`model/models.py`](../model/models.py) as
they are actually wired by the training scripts. Shapes use `B` for batch, `T`
for time (50 after `max_pad=True`), and `F` for a modality's feature width.

## `MultiFramework`

[`model/train_and_test.py`](../model/train_and_test.py) wraps three pieces:

1. **Encoders** — one `nn.Module` per modality, stored in `nn.ModuleList`.
2. **Fusion** — consumes the list of encoder outputs.
3. **Head** — maps the fused vector to a scalar sentiment score.

```python
outs = [encoders[i](inputs[i]) for i in range(n)]
fused = fusion(outs)
y_hat = head(fused)          # typically [B, 1]
```

When `has_padding=True` (`is_packed=True` in `train` / `test`), each encoder
receives `[tensor, lengths]` so LSTM/GRU can call `pack_padded_sequence`.

GMTM training scripts set every encoder and the head to `Identity` and put the
entire network inside the fusion slot:

```python
encoders = [Identity(), Identity(), Identity()]
fusion = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=HParams)
head = Identity()
```

## Gated Multi-Transformer Model (GMTM)

Class: `GatedMultiTransfomerModel`.

Constructor arguments:

- `n_modalities` — always `3` in the published scripts, even for ablations.
  Missing modalities are **zero tensors** of the original width, not omitted
  modules. See [`docs/datasets.md`](datasets.md#modality-ablation).
- `n_features` — list of input widths, visual / audio / text in that order:
  `[35, 74, 768]` (BERT) or `[35, 74, 300]` (GloVe).
- `hyp_params` — see [`hyperparameters.md`](hyperparameters.md).

### 1. Per-modality projection

Each modality `i` is mapped independently:

```
Linear(F_i → embed_dim) → LayerNorm → Dropout(embed_dropout) → ReLU
```

The forward pass reshapes `[B, T, F_i]` to `[T, B, embed_dim]` after this block
(`permute(1, 0, 2)` then a fused `T*B` linear).

### 2. Pairwise cross-modal transformers

For every ordered pair `(i, j)` a `TransformerEncoder` runs with

- query = projected modality `i`
- key = value = projected modality `j`

That is `n_modalities²` encoders (9 for the default three-way setup), plus a
`trans_mems` list that is constructed but **not** applied in the current
forward (the `self.trans_mems[i](h_list)` call is commented out).

Each `TransformerEncoder` adds sinusoidal positional embeddings, optional
future-masking (`attn_mask=True` in the training `HParams`), pre-norm
multi-head attention, and a 4× feed-forward block.

Text uses a slightly higher attention dropout
(`attn_dropout_modalities = [0, 0, 0.1]` for visual, audio, text).

### 3. Softmax modality weights

The `n` cross-attended sequences that target modality `i` are stacked and
reduced with a single softmax vector `modal_weights ∈ R^{n}` shared across
modalities (`nn.Parameter(torch.ones(n))` at init, so the softmax starts
uniform).

### 4. Sigmoid gate

```
h_i ← σ(W_i h_i) ⊙ h_i
```

One `Linear(embed_dim, embed_dim)` per modality. This is the "gated" part of
GMTM: a learned per-channel keep/drop after fusion.

### 5. Attention pooling and classification head

The gated sequences are concatenated on the feature axis to
`[B, T, embed_dim * n]`. `AttentionPooling` scores every time step with a
linear map to a scalar, softmaxes over `T`, and returns the weighted sum.

The head is:

```
LayerNorm → Linear(combined, combined/2) → ReLU → Dropout → Linear(..., 1)
```

`alpha` is registered as a learnable scalar but is unused in the current
forward (a residual mix was sketched in comments and left disabled).

### Default training `HParams` (BERT and GloVe GMTM scripts)

| Field | Value |
| --- | ---: |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `embed_dropout` | 0.2 |
| `attn_dropout` | 0.1 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |
| `attn_mask` | True |

The class-level `DefaultHyperParams` (`embed_dim=9`, `num_heads=3`, `layers=3`)
is **not** what the experiment scripts use.

## Fusion baselines

These are the six methods in `train_main_bert.py` / `train_main_glove.py`.

### Early concatenation — `ConcatEarly`

Stacks raw sequences on the feature axis: `[B, T, F_v + F_a + F_t]`.
BERT width is `35+74+768 = 877`; GloVe is `409`.

A packed LSTM then an MLP produce the scalar. Encoders are `Identity`.
Requires the *unpadded* loader (`max_pad=False`) so lengths are available.

### Late concatenation — `ConcatLate`

Independent LSTMs (BERT: 35→64, 74→256, 768→1024; GloVe text LSTM is 300→512)
flatten their last hidden states; `ConcatLate` cats them (`1344` or `832`
dims) into an MLP.

### Tensor Fusion — `TensorFusion`

Implements the outer-product fusion from
[Tensor Fusion Networks](https://github.com/Justin1904/TensorFusionNetworks).
Each modality is augmented with a trailing `1`, then successive einsum
products grow the last axis. BERT head input is `128000`; GloVe is `64000`.
Encoders are `GRUWithLinear` with reduced output widths (19 / 39 / 159 on BERT).

### Low-rank tensor fusion — `LowRankTensorFusion`

Factorized tensor fusion
([Low-rank Multimodal Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)).
Each modality has a factor of shape `(rank, input_dim+1, output_dim)`. The
scripts use `rank=32` and output dim `256` (BERT) or `128` (GloVe).

The ones-vector used to append the homogeneous coordinate is moved to
`cuda:0` when CUDA is available, otherwise CPU.

### Early transformer — `EarlyFusionTransformer`

`1×1` conv projects the concatenated features (`n_features=877` or `409`) to
`embed_dim=32`, a 4-layer `TransformerEncoder` (4 heads) runs over time, and
the **last time step** is returned. Head is `MLP(64, 64, 1)` on BERT and
`Identity` on GloVe.

Uses the **padded** loader (`max_pad=True`) and `is_packed=False`.

Note: the encoder layer is constructed with `batch_first=True`, but `forward`
permutes to `[T, B, 32]` before calling the encoder. The training scripts
rely on this as-is.

### Late transformer — `LateFusionTransformer`

Each modality is encoded with `TransformerSeq` (conv 1×1 + 4-layer encoder,
full sequence out). Sequences are concatenated on the feature axis
(BERT `64+128+1024 = 1216`, GloVe `64+128+512 = 704` in the encoder outputs;
the fusion module's `in_dim` is `1216` / `1792` in the two scripts) and
another transformer reads the joint sequence. The last step is scored by
`MLP(32, 32, 1)`.

This is the strongest baseline on MOSEI BERT (`MAE 0.5846`, Acc-2 `0.8393`).

### Extra module: `TransformerFusion`

A small encoder over a list of *already pooled* `[B, d]` vectors (stack as a
length-`n` sequence, mean-pool). It is defined in `models.py` but not used by
the main experiment scripts. The examples call it so the shape contract is
documented.

## Encoder building blocks

| Class | Behavior |
| --- | --- |
| `Identity` | Returns the tensor unchanged (or `x[0]` is not special-cased). |
| `Transformer` | Conv1d + 4-layer encoder; returns **last** time step. |
| `TransformerSeq` | Same stack; returns the **full** `[B, T, dim]` sequence. |
| `LSTM` | Optional packed sequence; last hidden, optional linear. |
| `GRU` / `GRUWithLinear` | Same idea with an optional projection. |
| `MLP` | `Linear → ReLU → Dropout → Linear`. |
| `Linear` (class) | Thin `nn.Linear` wrapper. Later in the file a **function** named `Linear` overwrites the class and applies Xavier uniform init. `TransformerEncoderLayer` uses the function. |
| `AttentionPooling` | Softmax-weighted sum over time. |
| `SinusoidalPositionalEmbedding` | Detached sinusoidal table, Fairseq-style `make_positions`. |

## Data flow in one training step

From `train()` in `train_and_test.py`, unpacked (`max_pad=True`) batch:

```
j = (vision, audio, text, labels)     # each [B, 50, F], labels [B, 1]
out = model([v.float(), a.float(), t.float()])
loss = L1Loss(out, labels)
clip_grad_norm_(..., 8)
AdamW step
```

Packed (`max_pad=False`) batch from `_process_1`:

```
(processed_input, lengths, inds, labels)
model([[v, a, t], lengths])
```

Validation keeps the checkpoint with the lowest mean loss. Early stopping
fires after **7** epochs without improvement when `early_stop=True`.
