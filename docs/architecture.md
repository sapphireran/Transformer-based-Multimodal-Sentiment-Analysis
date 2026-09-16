# Architecture notes

All neural modules live in [`model/models.py`](../model/models.py). Training
wraps them with [`MultiFramework`](../model/train_and_test.py), which is just:

```
encoders[i](modality_i)  →  fusion(list_of_reps)  →  head(fused)  →  ŷ
```

GMTM is special: its encoders are `Identity`, and the fusion module *is* the
full cross-modal transformer + classification head. The outer `head` is also
`Identity`, so GMTM's own `classification_head` is what produces the scalar.

## Feature convention

Unless a loader is in packed-length mode, each modality is

```
[batch, time, feat]     time ≤ 50 after max-pad
```

Word-aligned MOSI/MOSEI clips are truncated/padded to `max_seq_len=50`.
Feature widths used throughout the training scripts:

```
visual = 35     # FACET 4.2
audio  = 74     # COVAREP
text   = 768    # BERT   or  300  # GloVe 840B
```

`Affectdataset` can optionally z-normalize each clip independently
(per-feature mean/std over time). Ablation loaders keep the three slots but
replace unused modalities with zeros of the same shape — GMTM always sees
three tensors.

## Building blocks

### Sequence encoders

| Class | Input | Output | Notes |
| --- | --- | --- | --- |
| `Identity` | anything | same | Used when fusion consumes raw sequences |
| `LSTM` | `[B, T, F]` or packed `(seq, lengths)` | `[B, hid]` (last hidden, flattened) | `has_padding=True` uses `pack_padded_sequence` |
| `GRU` / `GRUWithLinear` | same | last hidden, optional linear | Tensor-fusion family |
| `Transformer` | `[B, T, F]` | `[B, dim]` last time step | 1×1 `Conv1d` then 4-layer encoder |
| `TransformerSeq` | `[B, T, F]` | `[B, T, dim]` | Same stack, keeps time |
| `MLP` | `[B, F]` | `[B, out]` | Two linear layers, ReLU, optional dropout |
| `AttentionPooling` | `[B, T, F]` | `[B, F]` | Learned time weights, softmax over `T` |

`Transformer` / `TransformerSeq` project `F → embed_dim` with a bias-free
`Conv1d(kernel_size=1)`, then run `nn.TransformerEncoder` (4 layers, 4 heads).

### Fusion modules

#### ConcatEarly

```
[B, T, F_v] ⊕ [B, T, F_a] ⊕ [B, T, F_t]  →  [B, T, F_v+F_a+F_t]
```

Requires aligned time. BERT width after concat is `35+74+768 = 877`.
GloVe width is `35+74+300 = 409`. The main BERT script then runs
`LSTM(877, 1024) → MLP(1024, 1024, 1)`.

#### ConcatLate

Each encoder returns a vector. Concatenate on the feature axis:

```
[B, d_v] ⊕ [B, d_a] ⊕ [B, d_t]  →  [B, d_v+d_a+d_t]
```

BERT late concat uses `64 + 256 + 1024 = 1344` then `MLP(1344, 1344, 1)`.

#### TensorFusion (TFN)

For each modality `m`, prepend a constant-1 column and form the outer product
across modalities:

```
z_i = [1; h_i]
z   = z_v ⊗ z_a ⊗ z_t
```

Implemented with a running einsum:

```python
fused = torch.einsum('...i,...j->...ij', m, mod)
m = fused.reshape([*nonfeature_size, -1])
```

The BERT configuration maps encoders to 19 / 39 / 159 so the fused width is
`20 * 40 * 160 = 128000` (the extra 1 comes from the bias column). That is
why the head is `MLP(128000, 2048, 1)` — it is the most parameter-heavy
baseline in the sweep.

#### LowRankTensorFusion (LMF)

Same multiplicative fusion, but each modality has a factor
`[rank, input_dim+1, output_dim]`. The forward pass is:

```
for each modality:
    h' = [1; flatten(h)]
    factor_m = h' @ W_m          # [B, rank, out]
    fused = fused * factor_m     # Hadamard over modalities
ŷ = w_rank · fused + b
```

BERT script: inputs `[32, 64, 256]`, output 256, rank 32.

Device note: the ones-vector is moved to `cuda:0` if CUDA is visible, else
CPU. Examples therefore work on CPU.

#### TransformerEarly

`EarlyFusionTransformer(n_features)`:

1. If the input is a list, `torch.cat(..., dim=2)` (same as ConcatEarly).
2. `Conv1d(n_features → 32)`.
3. 4-layer transformer, `d_model=32`, 4 heads.
4. Return the **last time step** `[B, 32]`.

`n_features` is 877 (BERT) or 409 (GloVe). The class sets
`batch_first=True` on the encoder layer but then feeds
`[T, B, 32]` — see [quirks](quirks.md).

#### TransformerLate

Per-modality `TransformerSeq` keeps time. Outputs are concatenated on the
feature axis (`64+128+1024 = 1216` for BERT, `64+128+512 = 704` wait —
the late fusion module is constructed with `in_dim=1216` for BERT and
`in_dim=1792` for GloVe). Then a second transformer (`LateFusionTransformer`)
projects to 32-d and returns the last step.

GloVe late path: `TransformerSeq(35,64) + TransformerSeq(74,128) +
TransformerSeq(300,512) → 704` but the script passes `in_dim=1792`. That
mismatch is recorded in [quirks](quirks.md); the published GloVe late numbers
were produced by whatever checkpoint was on disk, not by re-deriving this
arithmetic.

### Gated multi-transformer (GMTM)

Class name in code: `GatedMultiTransfomerModel` (typo preserved).

Default training hyperparameters (`HParams` in `train_GMTM_*.py`):

| Field | Value |
| --- | --- |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `attn_dropout` | 0.1 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |

Forward pass, for `n_modalities = 3`:

```
for i in {v, a, t}:
    h_i = Linear+LN+Dropout+ReLU (F_i → 64)     # applied per time step

for i in {v, a, t}:
    for j in {v, a, t}:
        h_ij = TransformerEncoder(q=h_i, k=h_j, v=h_j)   # cross-modal
    h_i' = Σ_j softmax(w)_j * h_ij                       # modality weights
    h_i' = σ(W_g h_i') ⊙ h_i'                            # gate

H = concat_i h_i'                         # [B, T, 192]
z = AttentionPooling(H)                   # [B, 192]
ŷ = LN → Linear(192, 96) → ReLU → Drop → Linear(96, 1)
```

`TransformerEncoder` here is the **custom** MulT-style stack (sinusoidal
positions, optional cross-attention when `x_in_k` / `x_in_v` are set), not
`nn.TransformerEncoder`.

`trans_mems` (self-attention memory) is constructed but the call is commented
out, so those parameters are unused during the forward that produced the
ablation CSV.

`alpha` is a leftover `nn.Parameter` from an abandoned residual around the
classification head. It is not applied.

## Shape cheat sheet (BERT, max-pad, batch 32)

| Stage | Shape |
| --- | --- |
| raw vision / audio / text | `[32, 50, 35]`, `[32, 50, 74]`, `[32, 50, 768]` |
| ConcatEarly fused | `[32, 50, 877]` |
| ConcatLate fused | `[32, 1344]` |
| LMF fused | `[32, 256]` |
| TFN fused | `[32, 128000]` |
| TransformerEarly last step | `[32, 32]` |
| GMTM after concat | `[32, 50, 192]` |
| GMTM after pooling / head | `[32, 192]` → `[32, 1]` |

`examples/inspect_shapes.py` prints these for random tensors on CPU.

## Complexity (qualitative)

- **TFN** blows up as `Π (d_i + 1)`. Fine for the reduced 19/39/159 setup,
  unusable if you fused raw 768-d BERT.
- **LMF** is the practical tensor-fusion baseline (rank 32).
- **TransformerLate** is the strongest non-GMTM head on the BERT MOSEI table.
- **GMTM** is pairwise `n²` cross-modal encoders (9 stacks × 4 layers). That
  is the reason the ablation script keeps `Identity` encoders — almost all
  capacity is already inside the fusion module.
