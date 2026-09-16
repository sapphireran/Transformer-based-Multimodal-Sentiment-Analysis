# Architecture notes

This file describes the modules in [`model/models.py`](../model/models.py)
and how [`MultiFramework`](../model/train_and_test.py) wires them. Shapes
below match the code as written, including a few quirks that are easy to
trip over when you first read the forward passes.

## The three-slot stack

Every trained experiment is the same sandwich:

```
x_v, x_a, x_t  -->  encoders[i]  -->  fusion  -->  head  -->  yhat (B, 1)
```

`MultiFramework.forward` walks the encoder list, stores the list on
`self.reps`, calls `self.fuse`, stores that on `self.fuseout`, then
applies `self.head`. Packed batches pass `(features, lengths)` into each
encoder; padded transformer batches pass a plain tensor per modality.

Identity encoders are used whenever the fusion module wants raw sequences
(GMTM, ConcatEarly, TransformerEarly). Recurrent encoders (LSTM / GRU)
are used when the fusion module wants a single vector per clip.

## Input ranks

| Path | Visual / audio / text tensor | Extra |
| --- | --- | --- |
| Padded (`max_pad=True`, `_process_2`) | `(B, 50, F)` | none |
| Packed (`max_pad=False`, `_process_1`) | `(B, T_max, F)` | lengths `(B,)` per modality |

`F` is 35 / 74 / 768 (BERT) or 35 / 74 / 300 (GloVe).

GMTM's first line is `x[i].permute(1, 0, 2)`, so it **requires**
batch-first sequences: `(B, T, F)` becomes `(T, B, F)` for the
transformer stack.

## Building blocks

### Sequence encoders

- `Transformer` — 1×1 `Conv1d` to `embed_dim`, 4-layer encoder, **last
  timestep only**.
- `TransformerSeq` — same stem, but returns the full `(B, T, embed_dim)`
  sequence. Used as the late-fusion unimodal encoder.
- `LSTM` / `GRU` / `GRUWithLinear` — optional packed sequences, optional
  dropout, optional projection. `has_padding=True` expects
  `[tensor, lengths]`.
- `MLP` — two linear layers with ReLU on the first. There is a latent
  bug in the second dropout branch (`dropout_layer(output)` instead of
  `output2`); the training scripts usually keep `dropout=False` on the
  head, so it does not fire on the recorded runs.
- `Identity` — pass-through, used as a dummy encoder or dummy head.

### Fusion operators

**ConcatEarly** (`ConcatEarly`) concatenates on dim 2:

```
(B, T, 35) + (B, T, 74) + (B, T, 768) -> (B, T, 877)
```

The BERT early-fusion head is then `LSTM(877, 1024) -> MLP(1024, 1024, 1)`.

**ConcatLate** (`ConcatLate`) flattens each encoder output from dim 1 and
concatenates on dim 1. After three LSTMs that is

```
64 + 256 + 1024 = 1344   (BERT)
64 + 256 + 512  = 832    (GloVe)
```

**TensorFusion** builds the multimodal tensor

```
z = [1; z_v] ⊗ [1; z_a] ⊗ [1; z_t]
```

Encoder output sizes on the BERT run are 19 / 39 / 159 so the fused
vector is `20 * 40 * 160 = 128000`. The GloVe run uses 19 / 39 / 79 →
`64000`. This is why the TensorFusion head is a wide MLP.

**LowRankTensorFusion** keeps a rank-`r` factor
`(rank, F_m + 1, out_dim)` per modality, multiplies the projected
`[1; z_m]` slices, and mixes ranks with `fusion_weights`. BERT uses
`[32, 64, 256] -> 256` at rank 32; GloVe uses `[32, 64, 128] -> 128`.

**EarlyFusionTransformer** concatenates sequences (same as ConcatEarly),
projects to `embed_dim=32` with a 1×1 conv, runs a 4-layer encoder, and
returns the last timestep `(B, 32)`. The module constructs an unused
`nn.Linear(32, 1)`; the actual scalar head sits outside, except in the
GloVe script where the head is `Identity` and the unused linear is still
not applied.

Note the `batch_first=True` flag on the encoder layer combined with a
seq-first permute in `forward`. PyTorch still runs; the flag and the
permute disagree. Treat the permute as the source of truth.

**LateFusionTransformer** concatenates already-encoded sequences on the
feature axis and runs the same 32-d encoder. BERT late fusion feeds
`64 + 128 + 1024 = 1216`; GloVe feeds `64 + 128 + 512 = 704` in the
module default path, but `train_main_glove.py` sets `in_dim=1792`
(`64+128+512` would be 704 — the 1792 value matches a wider text
encoder that was tried and left in the script). When you rerun, set
`in_dim` to the real concatenated width or the `Conv1d` will throw.

**TransformerFusion** (not used in the sweep scripts) stacks unimodal
vectors as a short "sequence" of modalities and mean-pools the encoder
output.

## GMTM (`GatedMultiTransfomerModel`)

GMTM is the custom model. A clip is three sequences. Each modality is
projected to a shared `embed_dim` (64 in the recorded runs):

```
Linear(F_m, D) -> LayerNorm -> Dropout -> ReLU
```

For every ordered pair `(i, j)` a `TransformerEncoder` attends from
modality `i` (query) into modality `j` (key / value). That is a full
`n × n` grid, including self-attention on the diagonal.

The `n` pair outputs for target modality `i` are mixed with a softmax
over a learned `modal_weights` vector of length `n` (the same weights
for every target). A per-modality linear + sigmoid then gates the mixed
sequence:

```
h_i = σ(W_i h̃_i) ⊙ h̃_i
```

The gated sequences are concatenated on the feature axis
`(B, T, n·D)` and collapsed with `AttentionPooling` (a linear score per
timestep, softmax over `T`, weighted sum). A two-layer MLP produces the
scalar.

`trans_mems` (a second transformer per modality) is constructed but the
call is commented out. `self.alpha` is a leftover residual scalar and is
not used in the return path.

### GMTM shapes (BERT, default HParams)

| Stage | Shape |
| --- | --- |
| Input `x[i]` | `(B, 50, F_i)` |
| After `modal_proj` | `(50, B, 64)` |
| After pair transformer `trans[i][j]` | `(50, B, 64)` |
| After weight + gate | `(B, 50, 64)` |
| After concat | `(B, 50, 192)` |
| After attention pool | `(B, 192)` |
| After classification head | `(B, 1)` |

Cross-attention uses `TransformerEncoder` with sinusoidal positions and
optional key/value inputs. `attn_mask=True` in HParams, but
`create_attention_mask` currently returns `None`, so the mask is not
applied.

## Complexity notes

`all_in_one_train` / `all_in_one_test` wrap the loop with
`memory_profiler.memory_usage` and print wall time plus parameter
count. They do not change the math.

Approximate parameter pressure:

- Concat / LRF heads are small.
- TensorFusion's `MLP(128000, 2048, 1)` dominates the BERT baseline.
- GMTM's cost is the `n²` cross-modal encoders (9 stacks × 4 layers at
  `D=64`) plus the projections. It is cheaper than TensorFusion and
  more expensive than ConcatLate.

## Device

Training scripts call `.cuda()` on every module. The synthetic examples
pick `cuda` if it exists and otherwise stay on CPU. `LowRankTensorFusion`
builds its ones-vector on `cuda:0` when CUDA is visible, independent of
the incoming tensor's device — keep inputs and the module on the same
device if you move off the default GPU.

## What I would change next (personal)

These are notes, not patches:

1. Honor `attn_mask` or drop the flag so the HParams match the code.
2. Use the constructed `trans_mems` or delete them.
3. Make `LateFusionTransformer.in_dim` compute from encoder widths.
4. Fix the second dropout in `MLP`.
5. Stop calling `plt.show()` inside `single_test` so headless eval does
   not block.
