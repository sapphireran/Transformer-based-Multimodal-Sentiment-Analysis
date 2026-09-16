# Fusion methods

Every baseline in `train_main_bert.py` / `train_main_glove.py` is a triple
`(encoders, fusion, head)`. This page lists the fusion operators
themselves and the encoder / head pairing the scripts actually use.

Tensor layouts below use `B` = batch, `T` = time, `F` = feature width.
When a method flattens, `F` may already be a hidden size rather than the
raw FACET / COVAREP / BERT width.

## ConcatEarly

**Class:** `ConcatEarly`

**Operation:** `torch.cat(modalities, dim=2)` — concatenate on the
feature axis, keep time.

```text
[B, T, 35] , [B, T, 74] , [B, T, 768]  →  [B, T, 877]
```

The fusion module does no learned work. All capacity sits in the head,
which is `LSTM(877, 1024) → MLP(1024, 1024, 1)` for BERT and
`LSTM(409, 512) → MLP(512, 512, 1)` for GloVe. Encoders are `Identity`.

**When it helps:** the LSTM can model cross-modal timing because the
three streams share a clock. **When it hurts:** a 877-d (or 409-d) input
makes the recurrent layer wide and easy to overfit, and a weak modality
is free to pollute every timestep.

## ConcatLate

**Class:** `ConcatLate`

**Operation:** flatten each encoder output from dim 1, then
`torch.cat(..., dim=1)`.

```text
[B, 64] , [B, 256] , [B, 1024]  →  [B, 1344]     (BERT)
[B, 64] , [B, 256] , [B, 512]   →  [B, 832]      (GloVe)
```

Each modality is encoded independently with an LSTM (BERT hidden sizes
64 / 256 / 1024, GloVe 64 / 256 / 512). The head is a two-layer MLP of
the concatenated width.

**When it helps:** each encoder can specialize; a collapsed visual
vector cannot drown BERT at every timestep. **When it hurts:** all
cross-modal interaction is deferred to a shallow MLP, so the model
never sees "this frown happened on this word".

## TensorFusion

**Class:** `TensorFusion`

**Operation:** append a homogeneous coordinate `1` to each vector, then
fold the outer product across modalities:

```text
m ← [1 ; z_v]
m ← m ⊗ [1 ; z_a]
m ← m ⊗ [1 ; z_t]
```

For vectors of length `d_v, d_a, d_t` the result has length
`(d_v+1)(d_a+1)(d_t+1)`. The BERT script therefore *shrinks* the
encoder outputs to 19 / 39 / 159 so the fused size is `20*40*160 = 128000`.
GloVe uses 19 / 39 / 79 → `20*40*80 = 64000`. The head is
`MLP(128000, 2048, 1)` or `MLP(64000, 2048, 1)`.

This is the original TFN construction
([Zadeh et al., 2017](https://github.com/Justin1904/TensorFusionNetworks)).
The extra `1` is what lets the product contain unimodal and bimodal
slices, not only the full trimodal interaction.

**Cost:** the fused vector is huge. That is why Low-Rank Tensor Fusion
exists.

## LowRankTensorFusion

**Class:** `LowRankTensorFusion(input_dims, output_dim, rank)`

**Operation:** for each modality `i`, append a `1`, then multiply by a
learned factor `W_i ∈ R^{rank × (d_i+1) × output_dim}`. The `rank`
slices are multiplied elementwise across modalities and contracted with
a vector `fusion_weights ∈ R^{1 × rank}`.

```text
BERT:  input_dims = [32, 64, 256], output_dim = 256, rank = 32
GloVe: input_dims = [32, 64, 128], output_dim = 128, rank = 32
```

Encoders are `GRUWithLinear` so the sequence is already a vector before
fusion. The head is an MLP of `output_dim`.

On MOSEI-BERT this is the strongest *non-Transformer* baseline
(MAE 0.5970, Acc-2 0.8305). It keeps TFN's multiplicative interactions
without materializing the 128k-d tensor.

`LowRankTensorFusion.forward` builds its ones-vector on CUDA whenever
`torch.cuda.is_available()` is true. Run the module on the same device
as that vector.

## TransformerEarly

**Class:** `EarlyFusionTransformer(n_features)`

**Operation:**

1. If the input is a list, `torch.cat` on dim 2 (same as ConcatEarly).
2. `Conv1d(n_features, 32, kernel_size=1)` maps the concatenated width
   to a shared 32-d embedding.
3. A 4-layer, 4-head Transformer encoder runs over time.
4. The **last timestep** is returned (`x[-1]` after the `[T, B, 32]`
   permute).

`n_features` is 877 (BERT) or 409 (GloVe). The BERT script then applies
`MLP(64, 64, 1)` — note the 64 vs 32 mismatch; the GloVe script uses
`Identity` as the head, so the prediction is the 32-d vector reduced
only if a later change adds a linear map. When you retrain, make the
head width match `embed_dim` (32).

This method uses the **max-padded** dataloader (`max_pad=True`) so every
clip is a dense `[B, 50, F]` cube. `is_packed=False`.

## TransformerLate

**Class:** `LateFusionTransformer(in_dim)`

**Operation:**

1. If the input is a list, `torch.cat` on the last dim.
2. `Conv1d(in_dim, 32, kernel_size=1)`.
3. 4-layer Transformer over time; return the last timestep.

The *encoders* are the interesting part. Each modality is a
`TransformerSeq` that keeps time:

| Embedding | Visual | Audio | Text | Concatenated `in_dim` |
| --- | ---: | ---: | ---: | ---: |
| BERT | 35 → 64 | 74 → 128 | 768 → 1024 | 1216 |
| GloVe | 35 → 64 | 74 → 128 | 300 → 512 | 704? / 1792 in script |

The GloVe script passes `in_dim=1792` (`64+128+512*3` is not 1792;
`64+128+1600` is also not it). `64+128+512 = 704`. The 1792 value is
what the saved GloVe checkpoint was built with — if you instantiate a
fresh `LateFusionTransformer(1792)` against `TransformerSeq` outputs
that only sum to 704, the `Conv1d` will throw. When running the
synthetic demo we use the **true** concatenated width.

On MOSEI-BERT, TransformerLate is the best baseline (MAE 0.5846, Corr
0.7041), second only to GMTM.

## TransformerFusion (unused by the sweep scripts)

**Class:** `TransformerFusion(d_model, nhead, num_layers)`

Treats *modalities* as the sequence axis: stack `[n_mod, B, D]`, permute
to `[B, n_mod, D]`, run a Transformer encoder, mean-pool over
modalities, project with `Linear(D, D)`.

This is a clean "modality-as-token" fusion. None of the `train_main_*`
scripts instantiate it; GMTM is the production pairwise alternative.

## GMTM

GMTM is documented in [architecture.md](architecture.md). Relative to
the baselines:

- It is the only method that keeps **pairwise cross-attention** between
  every ordered pair of modalities.
- It fuses with a **learned softmax** over those pairs plus a
  **per-modality gate**, instead of a single concat or outer product.
- It pools time with attention rather than taking the last hidden state.

Encoders are `Identity`; the head is `Identity` because GMTM already
ends in a 1-d linear.

## Pairing cheat sheet (BERT / MOSEI)

| Fusion | Encoders | Packed? | Head in |
| --- | --- | --- | --- |
| ConcatEarly | 3 × Identity | yes | 1024 |
| ConcatLate | LSTM 64 / 256 / 1024 | yes | 1344 |
| LowRankTensorFusion | GRUWithLinear → 32 / 64 / 256 | yes | 256 |
| TensorFusion | GRUWithLinear → 19 / 39 / 159 | yes | 128000 |
| TransformerEarly | 3 × Identity | no (max-pad) | 64 |
| TransformerLate | TransformerSeq 64 / 128 / 1024 | yes | 32 |
| GMTM | 3 × Identity | no (max-pad) | identity |

"Packed" here means `is_packed=True` in `train()` / `test()`, which is
what `has_padding=True` encoders expect.

## Choosing a method

A practical order to try, given the recorded MOSEI-BERT numbers:

1. **GMTM** if you can afford the pairwise grid.
2. **TransformerLate** if you want a single extra Transformer after
   per-modality encoding.
3. **LowRankTensorFusion** if you want multiplicative interactions
   without a Transformer.
4. **ConcatLate** as a cheap, reliable baseline.
5. **ConcatEarly** / **TransformerEarly** to test whether raw-time
   concatenation is enough (usually it is not).
6. **TensorFusion** only if you have the memory for the full product
   and want a published-TFN comparison point.
