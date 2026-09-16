# Module reference

A class-by-class index of [`model/models.py`](../model/models.py) and
[`model/train_and_test.py`](../model/train_and_test.py). Shapes are the
ones the experiment scripts actually pass.

## `model/models.py`

### `Identity`

`forward(x) → x`. Used as a no-op encoder or head so fusion modules can own
the whole graph (GMTM) or so early fusion can see raw sequences.

### `Transformer`

- `Conv1d(n_features → dim, k=1)`
- `TransformerEncoder` × 4, 4 heads
- Input `[B, T, F]` (or a 1-element list); output **last step** `[B, dim]`

### `TransformerSeq`

Same stem as `Transformer` but returns the full sequence `[B, T, dim]`.
Used as the per-modality encoder for late transformer fusion.

### `LSTM`

`nn.LSTM(indim, hiddim, batch_first=True)`. Optional
`pack_padded_sequence` when `has_padding=True` (input is then
`[tensor, lengths]`). Returns the last hidden state, flattened, with
optional `Linear` and dropout.

### `GRU` / `GRUWithLinear`

GRU analogue. `GRUWithLinear` projects `hiddim → outdim`.
`output_each_layer=True` returns a MultiBench-style list
`[0, flat_x, flat_h, leaky(out)]` (unused by the current scripts).

### `MLP`

Two-layer perceptron, ReLU after the first layer, optional dropout on both.
`output_each_layer=True` returns intermediate activations.

### `Linear` (class) and `Linear` (function)

The class is a thin `nn.Linear` with optional Xavier on the *class* path
(`nn.init.xavier_normal`). Later in the file a **function** `Linear` shadows
it and uses `xavier_uniform_` plus zero bias. `TransformerEncoderLayer.fc1/fc2`
call the function.

Prefer `CustomLinear` if you import from outside and want an unambiguous
module.

### `TransformerFusion`

Input: `List[[B, d]]`. Stack → `[B, n, d]`, encoder, mean over `n`,
`Linear(d, d)`. Not wired into `train_main_*.py`.

### `ConcatLate`

Flattens each tensor from dim 1 and concatenates on dim 1.
Expects already-pooled vectors `[B, F_i]`.

### `ConcatEarly`

`torch.cat(modalities, dim=2)` — sequences must share `B` and `T`.

### `TensorFusion`

Homogeneous outer products across an arbitrary number of modalities.
One modality → identity. Output last-dim is `Π_i (F_i + 1)`.

### `LowRankTensorFusion`

`factors[i]`: `(rank, F_i+1, output_dim)`. Elementwise product of
`x̃_i @ factor_i` over modalities, then `fusion_weights @ · + bias`.
`flatten=True` collapses non-batch dims before the affine.

### `EarlyFusionTransformer`

Class attr `embed_dim = 32`. `Conv1d(n_features → 32)` + 4-layer encoder +
last time step. If `x` is a list, concatenates on dim 2 first.

### `LateFusionTransformer`

`Conv1d(in_dim → 32)` on the already-concatenated sequence, 4-layer encoder,
last step `[B, 32]`.

### `TransformerEncoder` / `TransformerEncoderLayer`

Custom stack (not `nn.TransformerEncoder`) used **inside GMTM**:

- scale by `√d`, add `SinusoidalPositionalEmbedding`
- optional cross-attention (`x_in_k`, `x_in_v`)
- pre-norm MHA + residual-ish sublayer + FFN (`4d`)
- `create_attention_mask` currently returns `None` (the `attn_mask` flag
  does not yet build a causal mask; `buffered_future_mask` exists but is
  unused)

### `SinusoidalPositionalEmbedding` / `make_positions`

Fairseq-style sinusoidal table. `forward` takes a `[B, T]` integer-like
tensor (GMTM passes `x.transpose(0,1)[:, :, 0]`, i.e. the first feature
channel as a dummy position source). Output is **detached**.

### `AttentionPooling`

`score = Linear(F → 1)`, softmax over `T`, weighted sum → `[B, F]`.

### `GatedMultiTransfomerModel`

Documented in [`architecture.md`](architecture.md). Public pieces:

- `modal_proj`, `trans` (n×n), `trans_mems`, `modal_weights`
- `gating_linears`, `attn_pooling`, `classification_head`
- `forward(list[[B, T, F_i]]) → [B, output_dim]`

### `CustomLinear` / `CustomLayerNorm`

Unambiguous wrappers; not referenced by the training scripts.

### `fill_with_neg_inf` / `buffered_future_mask`

Helpers for a triangular future mask. Ready for use; GMTM does not call
them today.

## `model/train_and_test.py`

| Symbol | Role |
| --- | --- |
| `eval_affect` | Acc-2 + binary F1, optional drop-zeros |
| `MultiFramework` | encoder / fusion / head container |
| `deal_with_objective` | dtype / squeeze adapter for losses |
| `getallparams` | parameter count |
| `all_in_one_train` / `all_in_one_test` | time + RAM wrappers |
| `train` | full fit loop + `torch.save` |
| `split_uniform_7` / `split_uniform_5` | equal-width binning on [−3, 3] |
| `single_test` | MAE / MSE / Corr / Acc-7 / Acc-5 / Acc-2 / F1 + confusion plot |
| `test` | complexity wrapper around `single_test`, then a second pass |

## `model/data/get_dataloader.py`

| Symbol | Role |
| --- | --- |
| `drop_entry` | drop empty-text rows |
| `z_norm` | per-row mean/std, first 50 steps |
| `Affectdataset` | slice / pad / optional z-norm |
| `get_dataloader` | train, valid, test |
| `get_ablation_dataloader` | same, with zeroed modalities |
| `get_mosi_dataloader` | merged-split eval loader |
| `get_ablation_mosi_dataloader` | merged + zeroed |
| `_process_1` / `_process_2` | variable vs fixed-length collate |

## Script entry points

| File | Embedding | What it runs |
| --- | --- | --- |
| `train_main_bert.py` | BERT | 6 fusion baselines on MOSEI |
| `train_main_glove.py` | GloVe | 6 fusion baselines on MOSEI |
| `train_GMTM_bert.py` | BERT | GMTM, optional 7-way ablation |
| `train_GMTM_glove.py` | GloVe | GMTM, 7-way ablation |
| `mosi_test/train_mosi_bert.py` | BERT | MOSEI baselines → MOSI |
| `mosi_test/train_mosi_glove.py` | GloVe | same (MultiBench import) |
| `mosi_test/mult_bert_mosi.py` | BERT | GMTM → MOSI |
| `mosi_test/mult_glove_mosi.py` | GloVe | GMTM ablation → MOSI |
