# Architecture

The code is a **toolkit of interchangeable pieces**, not a single named
network. Almost every experiment is:

```text
[visual seq] --> encoder_v --\
[audio seq]  --> encoder_a ---> fusion --> head --> scalar sentiment
[text seq]   --> encoder_t --/
```

`MultiFramework` in `train_and_test.py` is the glue: it owns the encoder
list, the fusion module, and the head. GMTM is the exception: fusion and head
live inside `GatedMultiTransfomerModel`, so the outer encoders and head are
`Identity`.

## Modalities and default widths

| Stream | Source in the pickles | Width (BERT setup) | Width (GloVe setup) |
| --- | --- | --- | --- |
| Visual | Facet 4.2 facial AUs / expressions | 35 | 35 |
| Audio | COVAREP acoustic descriptors | 74 | 74 |
| Text | word-aligned BERT or GloVe | 768 | 300 |

Sequences are word-aligned. Loaders truncate to `max_seq_len=50` when
`max_pad=True`. Packed loaders keep native lengths and pad only inside the
batch (`_process_1`).

A typical padded batch looks like:

```text
vision : (B, 50, 35)
audio  : (B, 50, 74)
text   : (B, 50, 768)   # or (B, 50, 300)
label  : (B, 1)         # continuous score
```

A packed batch is:

```text
(features, lengths, index, label)
features[i] : (B, T_max_in_batch, F_i)
lengths[i]  : (B,)
```

`LSTM` / `GRUWithLinear` honor `has_padding=True` by calling
`pack_padded_sequence`. Transformer-early does **not** use packing; it wants
the four-tensor padded layout.

## Building blocks in `models.py`

### Sequence encoders

| Class | Behavior | Typical use |
| --- | --- | --- |
| `Identity` | returns input unchanged | early fusion, GMTM |
| `LSTM` | last hidden state, optional linear | late concat |
| `GRU` / `GRUWithLinear` | GRU, optional projection | tensor fusion / LMF |
| `Transformer` | 1×1 conv to `dim`, 4-layer encoder, **last timestep** | compact unimodal encode |
| `TransformerSeq` | same stack, returns full `(B, T, dim)` | late transformer |

`Transformer` / `TransformerSeq` project with `Conv1d` (`kernel_size=1`)
before a vanilla `nn.TransformerEncoder` (`nhead=4`, `num_layers=4`).

`LSTM.forward` with `has_padding=True` expects `[tensor, lengths]`. Without
padding it accepts `(B, T, F)` or unsqueezes a 2-D tensor.

### Feed-forward

`MLP(indim, hiddim, outdim)` is ReLU → optional dropout → linear. The
sentiment head is almost always `outdim=1` because training uses `L1Loss` on
the raw score.

There is both a `Linear` **class** and a later `Linear(...)` **factory** in
the same file. The factory wins at import time and is what
`TransformerEncoderLayer` uses (Xavier-uniform weights). Do not instantiate
the class after import.

### Fusion modules

See [fusion_methods.md](fusion_methods.md) for experiment wiring. The
primitives are:

- `ConcatEarly` — `torch.cat(..., dim=2)` on aligned sequences
- `ConcatLate` — flatten each encoder output, `cat` on `dim=1`
- `TensorFusion` — TFN-style successive outer products with a leading 1
- `LowRankTensorFusion` — factorized LMF with rank `r` and output dim
- `TransformerFusion` — stack modality vectors as a short sequence
- `EarlyFusionTransformer` — concat features first, then a 4-layer encoder
- `LateFusionTransformer` — concat already-encoded sequences, then encode

`LowRankTensorFusion` appends a constant-1 column per modality (the TFN
homogeneous trick) and multiplies rank-`r` factors. Device for the ones
vector is `cuda` if available, otherwise CPU.

### Attention utilities

`TransformerEncoder` / `TransformerEncoderLayer` are a custom stack with
sinusoidal positions (`SinusoidalPositionalEmbedding`), optional
cross-attention (`x_in_k`, `x_in_v`), and pre-norm. GMTM instantiates one of
these networks for **every ordered pair** of modalities, plus a “memory”
network per modality (the memory path is constructed but the call is
commented out in `forward`).

`AttentionPooling` scores each timestep with a linear `F → 1` map, softmaxes
over time, and returns a weighted sum. GMTM uses it after concatenating the
three gated streams.

## `MultiFramework`

```python
outs = [encoders[i](inputs[i]) for i in range(n)]
fused = fusion(outs)
return head(fused)
```

When `has_padding=True`, each encoder receives `[inputs[0][i], inputs[1][i]]`
— the i-th modality tensor plus its length vector.

The module stores `reps` (pre-fusion encoder outputs) and `fuseout` so a
custom objective can regularize intermediate tensors. The default `L1Loss`
path ignores those caches.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel(n_modalities, n_features, hyp_params)` is the
custom model used in the ablation scripts.

Default hyperparameters in the training scripts (`HParams`):

| Field | Value |
| --- | --- |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `attn_dropout` | 0.1 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` |
| `relu_dropout` / `res_dropout` | 0.1 |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |
| `attn_mask` | `True` |

Forward pass, with `n_modalities = 3`:

1. **Project.** Each `x[i]` is `(B, T, F_i)`. GMTM permutes to `(T, B, F)`,
   applies `Linear → LayerNorm → Dropout → ReLU` to `embed_dim`, and keeps
   `(T, B, 64)`.
2. **Pairwise cross-attention.** For every target `i` and source `j`,
   `trans[i][j](proj_i, proj_j, proj_j)` lets modality `i` attend to `j`.
3. **Softmax modality weights.** The three (`i`-centered) results are stacked
   and contracted with `softmax(modal_weights)`.
4. **Gate.** `sigmoid(Linear(h)) * h` per target modality.
5. **Concat + pool.** Streams are concatenated on the feature axis
   (`3 * 64 = 192`) and attention-pooled over time.
6. **Head.** LayerNorm → Linear `192 → 96` → ReLU → Dropout → Linear `96 → 1`.

An `alpha` residual parameter is allocated but **not** applied; the residual
add is commented out.

Ablations do **not** shrink `n_modalities`. Missing streams are replaced with
zeros of the original shape (`get_ablation_dataloader`). GMTM still runs the
full 3×3 attention grid; a zeroed modality simply contributes a zero query or
key/value.

## Early vs late transformer layouts

`EarlyFusionTransformer`:

1. If the input is a list, `cat` on `dim=2` → `(B, T, sum F)`.
2. `Conv1d` projects `sum F → 32`.
3. A 4-layer encoder (`nhead=4`, `batch_first=True` in the layer ctor) is
   applied after a permute to `(T, B, 32)`.
4. The **last timestep** is returned as `(B, 32)`.

`LateFusionTransformer` does the same after the caller has already encoded
each stream (`in_dim` is the concatenated encoder width: `1216` for BERT late
fusion, `704` would match the GloVe `TransformerSeq` widths `64+128+512`; the
checked-in GloVe script passes `in_dim=1792`, which only matches if you change
those encoder sizes).

`TransformerEarly` in `train_main_bert.py` then applies `MLP(64, 64, 1)` even
though `EarlyFusionTransformer` emits 32-D vectors. That mismatch is in the
checked-in script; the synthetic example uses a 32-D head. If you revive
training, align those widths before running.

## Parameter scale (order of magnitude)

Rough counts for the BERT MOSEI recipes, useful when reading
`all_in_one_train` logs:

| Recipe | What dominates |
| --- | --- |
| ConcatEarly | LSTM on 877-D packed input, hidden 1024 |
| ConcatLate | Text LSTM `768 → 1024` |
| LowRankTensorFusion | Text GRU `768 → 1024 → 256` plus LMF factors |
| TensorFusion | Outer-product width `20*40*160 = 128000` MLP |
| TransformerEarly | 4-layer encoder on 32-D tokens |
| TransformerLate | Text `TransformerSeq` `768 → 1024` |
| GMTM | 3×3 custom encoders at `d=64`, plus the head |

TensorFusion is the memory outlier: each sample materializes a very wide
fused vector. Prefer LMF or GMTM if you are iterating on CPU.

## Design choices that affect results

- **Regression first.** Acc2 / Acc7 are post-hoc discretizations of a scalar.
  The model never sees a 7-way cross-entropy.
- **Text dominates.** Ablations show text-only GMTM already close to the
  full model; audio/visual-only runs collapse toward chance correlation.
- **Alignment is assumed.** `Affectdataset` trims leading zeros using the
  text stream when `aligned=True`. Empty text rows are dropped in
  `drop_entry`.
- **No official MOSI test-only protocol** in `get_mosi_dataloader` — it
  merges splits. Quote those numbers only as transfer diagnostics.
