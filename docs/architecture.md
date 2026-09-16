# Architecture

This note describes the modules in `model/models.py` and the `MultiFramework` wrapper in `model/train_and_test.py`. Shapes assume a batch of `B` clips, `T` aligned time steps (capped at 50), and per-modality widths `Dv=35`, `Da=74`, `Dt∈{768,300}`.

## Training wrapper

Every trainable experiment is:

```text
[visual, audio, text]  →  encoders[i]  →  fusion(list)  →  head  →  scalar score
```

`MultiFramework` stores the per-modality encoder outputs on `self.reps` and the fused tensor on `self.fuseout` so a custom objective can inspect them. Padding has two modes:

| Mode | When | Batch contract |
| --- | --- | --- |
| Packed (`is_packed=True`) | Concat / tensor / late transformer | `model([[x_v, x_a, x_t], lengths])` |
| Padded (`max_pad=True`) | Early transformer, GMTM | `model([x_v, x_a, x_t])` with `T=50` |

Packed encoders (`LSTM`, `GRUWithLinear`) call `pack_padded_sequence`. GMTM and the early transformer expect a dense `[B, 50, D]` cube.

## Encoder catalogue

| Class | Role | Typical use |
| --- | --- | --- |
| `Identity` | Pass-through | Early concat, early transformer, GMTM |
| `LSTM` | Last hidden state, optional linear | Late concat |
| `GRU` / `GRUWithLinear` | GRU then optional projection | Tensor fusion family |
| `MLP` | Two-layer ReLU MLP | Almost every head |
| `Transformer` | Conv1d → 4-layer encoder → last step | Standalone sequence encoder |
| `TransformerSeq` | Same stack, returns full `[B, T, dim]` | Late transformer unimodal towers |
| `AttentionPooling` | Softmax time weights | GMTM temporal pool |

`LSTM` / `GRUWithLinear` with `has_padding=True` take `(sequence, lengths)` rather than a raw tensor.

## Fusion family

### ConcatEarly

`torch.cat(modalities, dim=2)` on the time axis. BERT width becomes 877, GloVe 409. The head is usually `LSTM(width, hid) → MLP(hid, hid, 1)`. This is the cheapest multimodal baseline: no cross-term, no extra parameters in the fusion itself.

### ConcatLate

Each modality is encoded to a vector, flattened, then `torch.cat(..., dim=1)`. BERT late-concat head width is `64 + 256 + 1024 = 1344`. GloVe uses `64 + 256 + 512 = 832`.

### TensorFusion

Implements Zadeh et al. Tensor Fusion Networks. Each modality `m` is lifted to `[1; m]`, then the Kronecker product across modalities is taken with `einsum`. Three modalities of sizes `(19, 39, 159)` explode to `20 × 40 × 160 = 128000` (BERT) or `20 × 40 × 80 = 64000` (GloVe). The MLP head is correspondingly huge.

### LowRankTensorFusion

Low-rank factorization of the same outer product (Liu et al.). Each modality has a factor `P_i ∈ R^{rank × (d_i+1) × out}`. The forward pass is:

```text
z = 1
for m, P in zip(modalities, factors):
    z = z ⊙ ([1; flatten(m)] @ P)     # [rank, B, out]
y = fusion_weights @ z + bias
```

BERT uses output 256 / rank 32; GloVe uses output 128 / rank 32. This is the strongest non-transformer baseline on MOSEI BERT.

### TransformerEarly (`EarlyFusionTransformer`)

Concatenate features to `[B, T, 877]` (or 409), project with `Conv1d` to `embed_dim=32`, run a 4-layer `TransformerEncoder` (4 heads), keep the last time step. The module sets `batch_first=True` on the encoder layer but then permutes to `[T, B, D]` before calling it — the example suite still runs, but the batch/time axes are swapped relative to the flag. Treat the published checkpoint as the source of truth for this variant.

### TransformerLate (`LateFusionTransformer` + `TransformerSeq`)

Each modality is mapped by `TransformerSeq` to a shared-ish width (`64 / 128 / 1024` for BERT, `64 / 128 / 512` for GloVe), concatenated on the feature axis (`1216` or `704+… = 704?` — BERT is `64+128+1024=1216`, GloVe `64+128+512=704` but the script passes `in_dim=1792`; see [training](training.md)), then a second transformer collapses time.

On MOSEI BERT this is the best baseline (MAE 0.5846).

### TransformerFusion

A smaller helper: stack already-pooled modality vectors as a short sequence `[B, M, D]`, encode, mean-pool. Not used by the main train scripts; kept as a building block.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the custom model. Default hyperparameters live on `GatedMultiTransfomerModel.DefaultHyperParams`; the train scripts override them with a local `HParams` (4 heads, 4 layers, `embed_dim=64`).

```text
x_i: [B, T, D_i]
        │
        ▼
 Linear → LayerNorm → Dropout → ReLU     (modal_proj[i])
        │
        ▼
  h_ij = Trans_ij(q=x_i, k=x_j, v=x_j)   for all j   (pairwise MulT-style)
        │
        ▼
  h_i  = Σ_j softmax(w)_j · h_ij         (learnable modal_weights)
        │
        ▼
  h_i  = σ(W_i h_i) ⊙ h_i                (per-modality gate)
        │
        ▼
  concat_i h_i → AttentionPooling → MLP  (score)
```

### Pieces

1. **Unimodal projection.** Each modality is independently mapped to `embed_dim` so pairwise attention is legal. Projection is applied per time step after flattening `[T*B, D_i]`.
2. **Pairwise cross-modal transformers.** `self.trans[i][j]` is a `TransformerEncoder` that can take separate key/value streams (`x_in_k`, `x_in_v`). That is MulT-style directed attention: modality `i` queries modality `j`.
3. **Self / memory towers.** `self.trans_mems[i]` is constructed but the call is commented out (`#h_list = self.trans_mems[i](h_list)`).
4. **Softmax modality mix.** `modal_weights` is a length-`M` parameter; softmax mixes the `M` directed views of modality `i`.
5. **Gating.** A sigmoid linear layer per modality scales the fused sequence.
6. **Temporal pool.** `AttentionPooling` scores each time step with a linear unit and does a weighted sum.
7. **Head.** LayerNorm → Linear → ReLU → Dropout → Linear to `output_dim=1`.

`self.alpha` (residual mix) is allocated but unused; the residual block at the bottom of `forward` is commented out.

### Cross-attention implementation notes

`TransformerEncoder` adds sinusoidal positions (`SinusoidalPositionalEmbedding`) and optional pre-norm. `TransformerEncoderLayer.create_attention_mask` currently returns `None`, so `attn_mask=True` does not actually apply a future mask. `buffered_future_mask` exists but is unused.

There is a name clash in `models.py`: a `class Linear` is later overwritten by `def Linear(...)`, which is the Xavier factory used inside `TransformerEncoderLayer`. Import `models.Linear` and you get the factory, not the module class.

### Ablation protocol

`get_ablation_dataloader` does **not** shrink the GMTM to fewer towers. It keeps three slots and **zeros the dropped modalities**:

| Embedding | visual | audio | text |
| --- | --- | --- | --- |
| BERT | `(50, 35)` | `(50, 74)` | `(50, 768)` |
| GloVe | `(50, 35)` | `(50, 74)` | `(50, 300)` |

That is why GMTM is always constructed with `n_modalities=3` even for `['text']` only. A zero tensor still goes through `modal_proj` and pairwise attention, so “unimodal” runs are not a true single-tower model — they are a three-tower model with two silent inputs.

## Hyperparameters used in GMTM scripts

From `train_GMTM_bert.py` / `train_GMTM_glove.py`:

| Knob | Value |
| --- | --- |
| `num_heads` | 4 |
| `layers` | 4 |
| `embed_dim` | 64 |
| `attn_dropout` | 0.1 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` |
| `relu_dropout` / `res_dropout` | 0.1 |
| `out_dropout` | 0.1 |
| `embed_dropout` | 0.2 |
| `output_dim` | 1 |
| `modality_dropout` | 0.2 (declared, not read by GMTM) |

Combined classifier width is `embed_dim * n_modalities = 192`.

## Complexity (order-of-magnitude)

| Fusion | Extra params beyond encoders |
| --- | --- |
| ConcatEarly / ConcatLate | 0 (fusion is a cat) |
| TensorFusion | 0 in the fusion, huge MLP head |
| LowRankTensorFusion | `rank * Σ_i (d_i+1) * out + rank + out` |
| Early / late transformer | 4 × TransformerEncoderLayer(32 or embed) |
| GMTM | `M²` cross-modal encoders + `M` gates + pool + head |

GMTM is the heaviest: 3×3 = 9 transformer stacks of 4 layers at `d=64`. That is the point of the personal study — spend capacity on directed pairwise attention plus a learned gate, then see whether MOSEI BERT actually uses it (it does; GloVe is closer).
