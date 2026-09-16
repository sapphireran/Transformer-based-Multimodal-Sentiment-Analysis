# Architecture

This note describes the modules in [`model/models.py`](../model/models.py) as they are used
by the trainers. Names match the classes in that file.

## Input convention

A sample is three aligned sequences plus a scalar label:

```
vision: (T, 35)
audio:  (T, 74)
text:   (T, 768)   # BERT  or  (T, 300) for GloVe
label:  (1,)       # typically in [-3, 3]
```

`Affectdataset` with `max_pad=True` pads or truncates to `T = 50`.
`get_dataloader(..., max_pad=True)` therefore yields batches
`(vision, audio, text, y)` with shape `(B, 50, dim)`.

`max_pad=False` uses `_process_1`: variable-length padded batches plus length tensors.
Most LSTM/GRU fusion baselines set `is_packed=True` and consume those lengths.
Transformer-early and GMTM use `max_pad=True` and `is_packed=False`.

## Encoder + fusion + head

[`MultiFramework`](../model/train_and_test.py) is a three-stage wrapper:

1. **Encoders** — one module per modality (`Identity`, `LSTM`, `GRUWithLinear`, `TransformerSeq`).
2. **Fusion** — combines the encoder outputs into one vector (or a short sequence that a later pool reduces).
3. **Head** — maps the fused vector to a single regression score (`MLP` or `Identity` when the fusion module already has a classifier).

GMTM is the exception: encoders and head are `Identity`, and all cross-modal work lives inside
`GatedMultiTransfomerModel`.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel(n_modalities, n_features, hyp_params)` is the model this repo
is organized around.

Default experiment hyperparameters (from `train_GMTM_bert.py` / `mosi_test/mult_bert_mosi.py`):

| Field | Value | Meaning |
| --- | --- | --- |
| `embed_dim` | 64 | Shared per-modality width after the input projection |
| `num_heads` | 4 | Attention heads (must divide `embed_dim`) |
| `layers` | 4 | Depth of each pairwise `TransformerEncoder` |
| `attn_dropout_modalities` | `[0, 0, 0.1]` | Extra dropout when attending to vision / audio / text |
| `embed_dropout` | 0.2 | Dropout on the projected modality tokens |
| `output_dim` | 1 | Regression head |

Forward path, for modalities `i, j ∈ {vision, audio, text}`:

1. **Project** each `(B, T, F_i)` sequence to `(T, B, embed_dim)` with Linear + LayerNorm + Dropout + ReLU.
2. **Pairwise cross-attention.** `trans[i][j]` is a `TransformerEncoder` that takes query from
   modality `i` and key/value from modality `j`. This includes `i == j` (self-attention).
3. **Softmax modality weights.** The `n_modalities` pairwise outputs for a fixed `i` are stacked
   and reduced with a learned `modal_weights` vector.
4. **Gate.** `sigmoid(Linear(h)) * h` per modality.
5. **Concat** the gated streams along the feature axis → `(B, T, n_modalities * embed_dim)`.
6. **Attention pooling** over time (`AttentionPooling`) → `(B, combined_dim)`.
7. **Classification head:** LayerNorm → Linear → ReLU → Dropout → Linear → `(B, 1)`.

`trans_mems` (a second self-attention stack per modality) is constructed but the call is
commented out in `forward`. Residual mixing through `self.alpha` is also present as a
parameter but not applied.

Complexity is quadratic in both sequence length and number of modalities: there are
`n_modalities²` pairwise encoders, each with `layers` transformer blocks.

## Fusion baselines

These are the named methods in `train_main_bert.py` / `train_main_glove.py`.

### ConcatEarly

Concatenate the raw sequences on the feature axis (`35+74+768 = 877` for BERT),
then an LSTM + MLP head. Encoders are `Identity`. Uses packed sequences.

### ConcatLate

Independent LSTMs (`35→64`, `74→256`, `768→1024` for BERT), concatenate the last
hidden states (`1344-d`), then an MLP. This is the "late concat" / "LF-LSTM" style baseline.

MOSI transfer scripts name the same checkpoint `Concat` (see `train_mosi_bert.py`),
which loads `checkpoints/ConcatLate.pt` is **not** automatic — the MOSI BERT script
looks for `../checkpoints/Concat.pt`. Keep that naming mismatch in mind when you
copy weights; see [`mosi.md`](mosi.md).

### TensorFusion

Each modality is mapped by `GRUWithLinear` to a compact vector. `TensorFusion` then
forms the outer product of `[1; z_i]` across modalities (Zadeh et al., Tensor Fusion
Network). BERT dimensions in this repo are `(19, 39, 159)` → a `128000-d` fused vector.

### LowRankTensorFusion

Same idea with a rank-`32` factorization (`LowRankTensorFusion([32, 64, 256], 256, 32)`
on BERT). Cheaper than the full tensor product and usually the second-best non-transformer
baseline on MOSEI BERT in the checked-in tables.

### TransformerEarly

`EarlyFusionTransformer`: 1×1 conv from the concatenated feature width (`877` BERT,
`409` GloVe) down to `embed_dim=32`, a 4-layer transformer, last timestep as the fused
vector. Head is an MLP `32→64→1` (BERT) or `Identity` (GloVe script). Requires `max_pad=True`.

### TransformerLate

Per-modality `TransformerSeq` (`35→64`, `74→128`, `768→1024` BERT), concatenate the
per-timestep encodings (`1216-d` BERT, `704` wait — GloVe uses `64+128+512=704` into
`LateFusionTransformer(in_dim=1792)` in the GloVe trainer; the BERT trainer uses
`in_dim=1216`). A second transformer pools to one vector, then an MLP.

The GloVe `in_dim=1792` vs. encoder sum `704` is a historical mismatch in
`train_main_glove.py`. The BERT path is internally consistent (`64+128+1024=1216`).

## Shared building blocks

| Class | Use |
| --- | --- |
| `Identity` | Pass-through encoder / head |
| `MLP` | Two-layer ReLU MLP; optional per-layer dump |
| `LSTM` / `GRU` / `GRUWithLinear` | Recurrent encoders, optional packed lengths |
| `Transformer` / `TransformerSeq` | Conv1d projection + `nn.TransformerEncoder` |
| `TransformerEncoder` | Custom encoder with sinusoidal positions and optional cross-attn |
| `AttentionPooling` | Learned softmax weights over time |
| `ConcatEarly` / `ConcatLate` | Feature-axis vs. flattened-vector concat |
| `TensorFusion` / `LowRankTensorFusion` | Multilinear fusion |

## Device and dtype

Trainers call `.cuda()` on every module. `train()` / `single_test()` themselves
fall back to CPU when CUDA is missing, but the script-level `.cuda()` will fail
on a CPU-only box. Examples in [`examples/`](../examples) pick `cpu` or `cuda`
through `examples/common.py`.

## Related files

- Training loop: [`training.md`](training.md)
- Metrics: [`evaluation.md`](evaluation.md)
- MOSI transfer setup: [`mosi.md`](mosi.md)
