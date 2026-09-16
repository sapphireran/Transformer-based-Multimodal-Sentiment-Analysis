# Architecture

The training scripts never subclass a giant `LightningModule`. They build three
pieces, hand them to `MultiFramework`, and let `train()` own the loop.

```
inputs[i]  →  encoders[i]  →  fusion(list_of_reps)  →  head  →  scalar
```

That contract is what makes the fusion sweep in `train_main_bert.py` a
`if fusion_method == ...` block instead of six unrelated trainers.

## `MultiFramework`

Defined in `model/train_and_test.py`.

```
MultiFramework(encoders, fusion, head, has_padding=False)
```

- `encoders` is an `nn.ModuleList` with one entry per modality. Order is
  vision, audio, text.
- `fusion` consumes the list of encoder outputs.
- `head` maps the fused vector to a single logit / regression value.
- `has_padding` (the `is_packed` flag in `train()`) changes how inputs are
  sliced:
  - **packed:** `model([[mod0, mod1, mod2], lengths])` and each encoder
    receives `[tensor, length]`. This is the `LSTM` / `GRUWithLinear`
    `has_padding=True` path.
  - **unpacked / max-pad:** `model([vision, audio, text])` with fixed
    `T=50`. This is what `TransformerEarly` and GMTM use.

The module caches `self.reps` (pre-fusion) and `self.fuseout` (post-fusion)
so a custom objective could regularize them. The shipped scripts only pass
`L1Loss`, so those caches are unused at train time.

## Encoder catalogue

All of these sit in `model/models.py`.

| Class | Typical role | Output rank |
| --- | --- | --- |
| `Identity` | Early fusion: skip encoding | same as input |
| `LSTM` | Late concat / packed sequences | `(B, hid)` or `(B, linear_out)` |
| `GRU` / `GRUWithLinear` | TFN / LMF encoders | `(B, out)` when `has_padding` |
| `Transformer` | last-timestep transformer | `(B, dim)` |
| `TransformerSeq` | late transformer per modality | `(B, T, dim)` |
| `MLP` | regression head | `(B, 1)` |

`LSTM.forward` with `has_padding=True` uses `pack_padded_sequence` and then
takes the hidden state. With `has_padding=False` it accepts a plain
`(B, T, F)` tensor (and will unsqueeze a 2-d input).

`TransformerSeq` is the unimodal sequence encoder used before
`LateFusionTransformer`:

1. `Conv1d` projects `F → embed_dim` (kernel 1, no padding).
2. `nn.TransformerEncoder` with 4 layers, 4 heads.
3. Returns the **full** sequence `(B, T, embed_dim)`, not a pooled vector.

`Transformer` is the same stack but returns only the last timestep. GMTM does
**not** use these two classes; it has its own `TransformerEncoder`.

## Fusion sits between encoders and the head

See [fusion-methods.md](fusion-methods.md) for shapes. The six sweep methods
and GMTM plug into the same `MultiFramework` slot:

| Method | Encoders (BERT MOSEI) | Fusion | Head |
| --- | --- | --- | --- |
| ConcatEarly | 3× `Identity` | `ConcatEarly` | `LSTM(877→1024)` + `MLP` |
| ConcatLate | LSTM 35/74/768 → 64/256/1024 | `ConcatLate` | `MLP(1344→1)` |
| LowRankTensorFusion | `GRUWithLinear` → 32/64/256 | LMF rank 32, out 256 | `MLP(256→1)` |
| TensorFusion | `GRUWithLinear` → 19/39/159 | `TensorFusion` | `MLP(128000→1)` |
| TransformerEarly | 3× `Identity` | `EarlyFusionTransformer(877)` | `MLP(64→1)` |
| TransformerLate | `TransformerSeq` 64/128/1024 | `LateFusionTransformer(1216)` | `MLP(32→1)` |
| GMTM | 3× `Identity` | `GatedMultiTransfomerModel` | `Identity` (head is inside GMTM) |

GloVe swaps 768 for 300 and shrinks the LSTM / GRU / transformer widths to
match. Exact numbers: [hyperparameters.md](hyperparameters.md).

## Gated multi-transformer (GMTM)

`GatedMultiTransfomerModel` is the custom piece. The class name keeps the
original `Transfomer` spelling so checkpoints stay loadable.

### Hyperparameters

`GatedMultiTransfomerModel.DefaultHyperParams` (overridden by the `HParams`
objects in `train_GMTM_*.py`):

| Field | Default in class | Used in GMTM scripts |
| --- | ---: | ---: |
| `embed_dim` | 9 | 64 |
| `num_heads` | 3 | 4 |
| `layers` | 3 | 4 |
| `attn_dropout` | 0.1 | 0.1 |
| `attn_dropout_modalities` | zeros | `[0, 0, 0.1]` |
| `relu_dropout` / `res_dropout` | 0.1 | 0.1 |
| `embed_dropout` | 0.25 | 0.2 |
| `out_dropout` | 0.0 | 0.1 |
| `output_dim` | 1 | 1 |
| `attn_mask` | True | True |

`num_heads` must divide `embed_dim`. 4 divides 64; 3 divides 9. The toy
example uses a small legal pair so it runs on CPU.

### Forward pass

Inputs are a list of three tensors `x[i]` with shape `(B, T, F_i)`.

1. **Project.** Each modality goes through
   `Linear(F_i → embed_dim) → LayerNorm → Dropout → ReLU`.
   The linear is applied on a flattened `(T*B, F_i)` view, then reshaped
   back to `(T, B, embed_dim)`.
2. **Pairwise cross-attention.** For every ordered pair `(i, j)` a
   `TransformerEncoder` runs with query = modality `i` and key/value =
   modality `j`. That is `n_modalities²` encoders (9 for three modalities).
   Same-modality pairs are just self-attention.
3. **Modality weights.** The `n` outputs that share query `i` are stacked
   and reduced with a softmax over a learnable `modal_weights` vector of
   length `n_modalities`. One set of weights is shared across query
   indices.
4. **Gate.** `h = σ(W_i h) ⊙ h` with a per-modality linear map.
5. **Concat over modalities.** `(B, T, embed_dim)` tensors are concatenated
   on the last axis → `(B, T, n*embed_dim)`.
6. **Attention pooling.** `AttentionPooling` scores each timestep with a
   linear `n*embed_dim → 1`, softmaxes over `T`, and returns `(B, n*embed_dim)`.
7. **Head.** LayerNorm → Linear → ReLU → Dropout → Linear → `(B, 1)`.

A residual `alpha` parameter is constructed but the residual add is commented
out; the shipped forward returns the head output only.

`trans_mems` (a second transformer per modality) is built in `__init__` and
never called. It remains so old checkpoints that contain those weights still
load.

### Why GMTM is a fusion module, not a full model

`train_GMTM_bert.py` sets `encoders = [Identity, Identity, Identity]` and
`head = Identity`. All of the work is inside the fusion slot. That lets the
same `train()` / `test()` functions evaluate GMTM next to ConcatLate without
a second trainer.

## Custom transformer internals

GMTM's `TransformerEncoder` is not `nn.TransformerEncoder`. It:

- scales embeddings by `sqrt(embed_dim)`
- adds `SinusoidalPositionalEmbedding`
- supports a cross-attention path (`x_in_k`, `x_in_v`)
- uses pre-norm (`normalize_before = True`)
- expands the FFN to `4 * embed_dim`

`attn_mask=True` attaches a final `LayerNorm` and *requests* a future mask,
but `create_attention_mask` currently returns `None`. The flag therefore
only changes whether the encoder applies that last LayerNorm.

## Device and dtype

Scripts call `.cuda()` on every module. The examples resolve

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

and keep modules + tensors on that device. `LowRankTensorFusion` internally
builds a ones-vector on `cuda:0` if CUDA is visible, otherwise CPU — so a
CPU-only machine works as long as the inputs are CPU tensors.

## Related reading inside this repo

- Packed vs max-pad batch layout: [training.md](training.md)
- Exact fusion algebra: [fusion-methods.md](fusion-methods.md)
- Modality dropout via zeros: [datasets.md](datasets.md#ablation-zero-out)
