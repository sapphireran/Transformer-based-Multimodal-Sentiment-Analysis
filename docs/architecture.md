# Architecture

All trainable pieces live in [`model/models.py`](../model/models.py).
The training scripts wrap them in `MultiFramework`
(encoders → fusion → head) from
[`model/train_and_test.py`](../model/train_and_test.py). GMTM is the
exception: the fusion module already contains the classification head,
so the encoders and head are `Identity`.

## Modality order

Every public entry point uses this order:

1. **visual** — Facet 4.2, width 35
2. **audio** — COVAREP, width 74
3. **text** — BERT 768 or GloVe 300

A clip is a list of three tensors `[visual, audio, text]`, each
`[batch, time, feat]`. With `max_pad=True` the time axis is fixed at 50
(or `max_seq_len`). With `max_pad=False` the collate function pads to
the longest clip in the batch and also returns lengths.

## Building blocks

### Sequence encoders

| Class | Input | Output | Notes |
| --- | --- | --- | --- |
| `Identity` | anything | same | Used when fusion sees raw features |
| `LSTM` | `[B, T, F]` or packed `(data, lengths)` | `[B, hid]` (last hidden, flattened) | `has_padding=True` uses `pack_padded_sequence` |
| `GRU` / `GRUWithLinear` | same | last hidden, optional linear | Low-rank / tensor fusion encoders |
| `MLP` | `[B, F]` | `[B, out]` | Two-layer ReLU MLP; used as heads |
| `Transformer` | `[B, T, F]` | `[B, dim]` | Conv1d to `dim`, 4-layer encoder, **last timestep** |
| `TransformerSeq` | `[B, T, F]` | `[B, T, dim]` | Same stack but keeps the time axis |

`Transformer` / `TransformerSeq` project with a bias-free `Conv1d`
(`kernel_size=1`) so the transformer always sees `embed_dim` channels.
The encoder uses 4 heads and 4 layers (`nn.TransformerEncoderLayer`).

### Fusion modules

#### ConcatEarly

```python
torch.cat(modalities, dim=2)   # [B, T, F_v + F_a + F_t]
```

The head is usually `LSTM(total_dim, 1024) → MLP(1024, 1024, 1)` for
BERT (total 877) or `LSTM(409, 512) → MLP` for GloVe.

#### ConcatLate

Each encoder already collapsed time. Fusion flattens whatever is left
and concatenates on dim 1:

```python
torch.cat([m.flatten(1) for m in modalities], dim=1)
```

BERT late-concat uses LSTM widths 64 / 256 / 1024 → 1344-D MLP.

#### TensorFusion

Implements the TFN outer-product trick: prepend a 1 to each modality
vector, then iteratively `einsum('...i,...j->...ij')`. Three vectors of
size `(d+1)` become a `(d_v+1)(d_a+1)(d_t+1)` tensor. The BERT script
uses 19 / 39 / 159 so the fused width is 128000, then `MLP(128000, 2048, 1)`.

#### LowRankTensorFusion

Factorized TFN. Each modality gets a factor of shape
`[rank, input_dim+1, output_dim]`. The forward pass multiplies the
projected tensors across modalities and mixes ranks with
`fusion_weights`. BERT config: inputs `[32, 64, 256]`, output 256, rank 32.

#### TransformerEarly

Cat on the feature axis, `Conv1d` to 32 channels, 4-layer transformer,
return the last time step (`[B, 32]`). `n_features` is 877 (BERT) or
409 (GloVe). The class stores `embed_dim = 32` as a class attribute.

#### TransformerLate

Per-modality `TransformerSeq` produces `[B, T, d_i]`. Those are
concatenated on the last dim (BERT: 64+128+1024 = 1216; GloVe:
64+128+512 = 704, but the GloVe script sets `in_dim=1792` — see the
caveat below) and sent through another Conv1d + transformer. Output is
the last time step, 32-D, then `MLP(32, 32, 1)`.

#### GatedMultiTransfomerModel (GMTM)

This is the custom block. Name is misspelled in the class
(`Transfomer`) on purpose — do not rename it or the checkpoints will
not unpickle.

Forward pass, for `n_modalities=3` and `embed_dim=64`:

1. **Project** each `[B, T, F_i]` with `Linear → LayerNorm → Dropout → ReLU`
   to `[T, B, embed_dim]`.
2. **Cross-modal transformers.** For every ordered pair `(i, j)` a
   `TransformerEncoder` attends from modality `i` (query) to modality
   `j` (key/value). That is a 3×3 grid of encoders.
3. **Softmax modality weights** (`self.modal_weights`) mix the three
   cross-attn outputs for source `i`.
4. **Gate** `σ(W h)` multiplies the mixed stream elementwise.
5. Concatenate the three gated streams on the feature axis →
   `[B, T, embed_dim * n]`.
6. **Attention pooling** over time (`AttentionPooling`) →
   `[B, embed_dim * n]`.
7. **Classification head**: LayerNorm → Linear → ReLU → Dropout →
   Linear to `output_dim=1`.

Default hyperparameters in the training scripts (`HParams`):

```
num_heads = 4
layers = 4
embed_dim = 64          # must be divisible by num_heads
attn_dropout = 0.1
attn_dropout_modalities = [0, 0, 0.1]
relu_dropout = 0.1
res_dropout = 0.1
out_dropout = 0.1
embed_dropout = 0.2
attn_mask = True
output_dim = 1
```

`get_network` builds a custom `TransformerEncoder` (sinusoidal
positions, optional future mask, pre-norm). Memory transformers
(`trans_mems`) are constructed but the residual call is commented out
in `forward`.

## Tensor-shape cheat sheet

Assume BERT, `B=4`, `T=50`.

| Stage | Shape |
| --- | --- |
| raw visual / audio / text | `[4, 50, 35]` / `[4, 50, 74]` / `[4, 50, 768]` |
| ConcatEarly fused | `[4, 50, 877]` |
| ConcatLate (after LSTMs) | `[4, 1344]` |
| LRTF fused | `[4, 256]` |
| TFN fused | `[4, 128000]` |
| TransformerEarly out | `[4, 32]` |
| TransformerSeq text | `[4, 50, 1024]` |
| GMTM projected stream | `[50, 4, 64]` |
| GMTM after pooling | `[4, 192]` |
| any head / GMTM logit | `[4, 1]` |

## Layout caveat

`EarlyFusionTransformer` and `LateFusionTransformer` construct
`TransformerEncoderLayer(..., batch_first=True)` but then permute the
input to `[T, B, E]` (the historical PyTorch default). The published
checkpoints were trained with that layout, so examples keep it. If you
rewrite these two classes, retrain — do not reuse the old `.pt` files.

The GloVe late-transformer script sets `LateFusionTransformer(in_dim=1792)`
while the three `TransformerSeq` widths sum to 704. That only works if
you change the encoder widths or the `in_dim`. The synthetic example
uses `in_dim = 64+128+1024` for BERT and `64+128+512` for GloVe so a
forward pass succeeds.

## Complexity notes

`all_in_one_train` / `all_in_one_test` wrap the loop with
`memory_profiler.memory_usage` and print wall time plus parameter
count. GMTM with `embed_dim=64`, 3 modalities, and a 3×3 encoder grid
is the heaviest graph in the repo; the toy example in
`examples/train_toy_gmtm.py` shrinks `embed_dim` and `layers` so it
finishes on CPU.
