# Architecture

The training scripts never call a fusion module in isolation. They always
build a `MultiFramework` from `model/train_and_test.py`:

```
inputs (vision, audio, text)  →  encoders[]  →  fusion  →  head  →  scalar
```

`GatedMultiTransfomerModel` is the exception that **is** the fusion *and* the
head. Those runs pass `Identity` encoders and an `Identity` head so the
framework still has three slots.

## MultiFramework

```
class MultiFramework(encoders, fusion, head, has_padding)
```

- `encoders` is an `nn.ModuleList` with one encoder per modality, in the
  loader order **visual, audio, text**.
- If `has_padding` is true (packed LSTM / GRU runs), each encoder receives
  `[tensor, lengths]`.
- Encoder outputs are stored on `self.reps`, the fused tensor on
  `self.fuseout` (the optional `objective_args_dict` path reads both).
- The head always emits a single sentiment score (shape `[B, 1]`).

Packed vs. padded is a **fusion-specific** choice:

| Fusion | `is_packed` / `has_padding` | Sequence handling |
| --- | --- | --- |
| ConcatEarly / ConcatLate / TF / LRTF | `True` | variable length + `pack_padded_sequence` |
| TransformerEarly / GMTM | `False` | `max_pad=True` in the dataloader, `T=50` |
| TransformerLate | `True` in the BERT/GloVe sweeps | transformers still see padded batches |

`train_main_bert.py` therefore builds **two** dataloader triples: a packed
one (`*_OT`) and a max-padded one (`*_TE`) used only by `TransformerEarly`.

## Per-modality encoders

Defined in `model/models.py`:

| Module | What it does | Typical use |
| --- | --- | --- |
| `Identity` | passthrough | early fusion, GMTM |
| `LSTM` | last hidden state, optional linear | ConcatEarly head, ConcatLate encoders |
| `GRU` / `GRUWithLinear` | last (or full) hidden, optional proj | TensorFusion, LowRankTensorFusion |
| `MLP` | two-layer ReLU MLP | almost every head |
| `Transformer` | 1×1 conv + 4-layer encoder, **last timestep** | unused in the current sweeps |
| `TransformerSeq` | same stack, **full sequence** out | TransformerLate encoders |

`LSTM` and `GRUWithLinear` accept `has_padding=True` and then expect the
`(data, lengths)` pair that `_process_1` produces.

Feature widths after encoding (BERT / MOSEI, from `train_main_bert.py`):

| Fusion | Visual | Audio | Text | Fused width into the head |
| --- | ---: | ---: | ---: | ---: |
| ConcatEarly | 35 (raw) | 74 (raw) | 768 (raw) | LSTM(877 → 1024) then MLP → 1 |
| ConcatLate | LSTM 64 | LSTM 256 | LSTM 1024 | 1344 |
| LowRankTensorFusion | GRU 32 | GRU 64 | GRU 256 | 256 (rank 32) |
| TensorFusion | GRU 19 | GRU 39 | GRU 159 | 20×40×160 = 128000 |
| TransformerEarly | raw | raw | raw | transformer `embed_dim=32`, MLP 64→1 |
| TransformerLate | T×64 | T×128 | T×1024 | late transformer 1216 → 32 |

GloVe runs shrink the text encoder (300-d input) and the fused widths
accordingly — see `train_main_glove.py`.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the model the ablation scripts train. The
class name keeps the original typo (`Transfomer`).

### Hyperparameters used in the scripts

`train_GMTM_bert.py` / `train_GMTM_glove.py` / `mosi_test/mult_*_mosi.py`
share this `HParams` block:

| Field | Value | Meaning |
| --- | --- | --- |
| `embed_dim` | 64 | shared width after `modal_proj` |
| `num_heads` | 4 | must divide `embed_dim` |
| `layers` | 4 | depth of each `TransformerEncoder` |
| `attn_dropout_modalities` | `[0, 0, 0.1]` | extra dropout on the text key/value side |
| `attn_dropout` | 0.1 | used by the (currently unused) memory stack |
| `embed_dropout` | 0.2 | after the per-modality projection |
| `out_dropout` | 0.1 | inside the classification head |
| `output_dim` | 1 | sentiment scalar |
| `attn_mask` | `True` | enables the final `LayerNorm` on the encoder |

`modality_dropout` and `use_text_transformer` are declared on `HParams` but
**not read** by `GatedMultiTransfomerModel.__init__`.

### Forward pass, one batch

Inputs are a list `x[i]` with shape `[B, T, F_i]` (visual, audio, text).

1. **Project.** Each modality is reshaped to `[T*B, F_i]`, sent through
   `Linear → LayerNorm → Dropout → ReLU`, then restored to `[T, B, 64]`.
2. **Pairwise cross-attention.** For every ordered pair `(i, j)` a
   `TransformerEncoder` is called as
   `trans[i][j](query=proj_x[i], key=proj_x[j], value=proj_x[j])`.
   That is a full `n × n` grid (9 encoders when `n=3`).
3. **Modality softmax.** The nine outputs for a fixed query `i` are stacked
   and reduced with `einsum('m,m...->...', softmax(modal_weights), ...)`.
   `modal_weights` is a single learnable vector of length `n`, shared across
   query slots.
4. **Gate.** `h = sigmoid(W_i h) ⊙ h` per query modality.
5. **Concat over modalities.** `[B, T, 64]` tensors become `[B, T, 192]`.
6. **Attention pooling.** A linear `192 → 1` produces timestep weights;
   the weighted sum is `[B, 192]`.
7. **Head.** `LayerNorm → Linear(192, 96) → ReLU → Dropout → Linear(96, 1)`.

`self.alpha` is allocated as a learnable scalar but the residual mix that
would have used it is commented out. `trans_mems` (self-attention “memory”
stacks) are built in `__init__` and never called.

### Custom transformer stack

GMTM does **not** use `nn.TransformerEncoder`. It uses the local
`TransformerEncoder` / `TransformerEncoderLayer` plus
`SinusoidalPositionalEmbedding`:

- Embeddings are scaled by `sqrt(embed_dim)` before the sinusoids are added.
- Pre-norm: each layer applies `LayerNorm` before attention.
- Cross-attention is the same `MultiheadAttention` module with `k`/`v`
  taken from another modality.
- `create_attention_mask` currently returns `None`; `attn_mask=True` only
  turns on the encoder-level `LayerNorm`.

`Linear` is defined twice in `models.py` (a class, then a factory function).
The factory wins at import time and is what the custom encoder uses
(Xavier-uniform weights, zero bias).

## Early / late transformer fusion (non-GMTM)

`EarlyFusionTransformer(n_features)`:

1. Optional `torch.cat` on dim 2 if the input is a list.
2. `Conv1d(n_features → 32, kernel 1)`.
3. 4-layer `nn.TransformerEncoder` (`nhead=4`, `batch_first=True` on the
   *layer* constructor).
4. Returns the **last timestep** (`x[-1]` after permuting to
   `[T, B, 32]`).

The layer is constructed with `batch_first=True` but the tensor is then
permuted to `[T, B, C]`. The module still runs; the flag and the permute
disagree. The personal examples print both layouts so this is visible.

`LateFusionTransformer(in_dim)` is the same pattern after the per-modality
`TransformerSeq` outputs have been concatenated on the feature axis
(`in_dim=1216` BERT, `1792` GloVe).

## Complexity notes

`train_and_test.all_in_one_train` wraps the loop in `memory_profiler` and
prints wall time, peak RSS, and parameter count. `TensorFusion` is the
outlier: the head is `MLP(128000, 2048, 1)` on BERT. Low-rank fusion exists
specifically to avoid that product.

GMTM’s cost is dominated by the **nine** depth-4 encoders, not the pooling
head. An `n`-modality generalization is already written (`n_modalities` and
`n_features` are constructor arguments) but every training script hard-codes
`n=3`.
