# Fusion methods

Every method below implements `forward(modalities) -> Tensor` and is dropped
into `MultiFramework.fuse`. Shapes assume BERT-MOSEI unless noted.
`B` is batch, `T` is time (variable or 50), `F_v=35`, `F_a=74`, `F_t=768`.

The runnable counterpart is `examples/run_fusion_forward.py`, which builds
each module on CPU and prints the output shape.

## ConcatEarly

```python
torch.cat(modalities, dim=2)   # (B, T, F_v + F_a + F_t)
```

Encoders are `Identity`, so fusion sees raw aligned sequences. The head is
responsible for the temporal model: on BERT-MOSEI that is
`LSTM(877, 1024, has_padding=True)` then `MLP(1024, 1024, 1)`.

GloVe uses `F_t=300` so the concatenated width is `35+74+300=409`, and the
LSTM shrinks to hidden size 512.

**When to use it:** cheapest multimodal baseline. It cannot down-weight a
noisy modality except through the later LSTM.

## ConcatLate

Each modality is encoded to a **vector** first (packed LSTM):

| Modality | BERT LSTM | GloVe LSTM |
| --- | --- | --- |
| Vision 35 | 64 | 64 |
| Audio 74 | 256 | 256 |
| Text | 1024 (from 768) | 512 (from 300) |

`ConcatLate` flattens each vector (already rank-2) and concatenates on
`dim=1`:

- BERT: `64+256+1024 = 1344`
- GloVe: `64+256+512 = 832`

The head is a two-layer `MLP` of that width. Late concat lets each encoder
choose its own hidden size, which is why text is much wider than vision.

## TensorFusion (TFN)

Implementation follows the original
[TensorFusionNetworks](https://github.com/Justin1904/TensorFusionNetworks)
outer-product-with-ones trick.

For two vectors `a ∈ R^{p}`, `b ∈ R^{q}`:

```
[1, a] ⊗ [1, b]  ∈  R^{(p+1)(q+1)}
```

Three modalities fold left to right:

```
m ← [1, vis]
m ← m ⊗ [1, aud]
m ← m ⊗ [1, txt]
```

BERT encoder widths are 19 / 39 / 159 so the fused size is
`(19+1)*(39+1)*(159+1) = 20*40*160 = 128000`. The head is
`MLP(128000, 2048, 1)` — by far the largest parameter block in the sweep.

GloVe uses 19 / 39 / 79 → `20*40*80 = 64000`.

If only one modality is passed, TFN returns it unchanged. Ablation scripts
do **not** drop the extra tensors; they zero them, so TFN still sees three
inputs.

## LowRankTensorFusion (LMF)

Low-rank factorization of the same tensor, following
[Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion).

For each modality `i` with input width `d_i` there is a factor
`U_i ∈ R^{rank × (d_i+1) × out}`. The forward pass:

1. Append a 1 to each flattened modality vector.
2. `modality_factor_i = [1, x_i] @ U_i` → `(rank, out)`
3. Hadamard-product the factors across modalities.
4. Mix ranks with `fusion_weights ∈ R^{1×rank}` and add `fusion_bias`.

BERT config in `train_main_bert.py`:

```
input_dims = [32, 64, 256], output_dim = 256, rank = 32
```

GloVe: `[32, 64, 128]`, output 128, rank 32.

LMF is the strongest non-transformer in the BERT-MOSEI table (MAE 0.5970),
and the strongest GloVe fusion (MAE 0.6174).

## TransformerFusion (generic)

`TransformerFusion(d_model, nhead, num_layers, dropout)` stacks modality
**vectors** as a short sequence of length `n_modalities`, runs a
`TransformerEncoder`, mean-pools, and applies a linear `d_model → d_model`.

None of the published `train_*.py` scripts instantiate this class. It is
kept as a building block; the sweep uses the early / late variants below
instead.

## EarlyFusionTransformer

Used as the **fusion** module with Identity encoders and `max_pad=True`
batches (`T=50` for every sample).

1. If the input is a list, `torch.cat(..., dim=2)` — same as ConcatEarly.
2. `Conv1d` projects `n_features → embed_dim=32`.
3. A 4-layer, 4-head `nn.TransformerEncoder` runs over time.
4. The last timestep `(B, 32)` is returned.

`n_features` is 877 (BERT) or 409 (GloVe). The class also constructs
`self.linear = nn.Linear(32, 1)` but `forward` does not call it; the
BERT script attaches `MLP(64, 64, 1)` as the `MultiFramework` head
instead. Treat that MLP width as part of the published recipe, not as a
shape that is guaranteed by `EarlyFusionTransformer` alone.

`batch_first=True` is set on the encoder layer while `forward` permutes to
`(T, B, 32)` before calling the encoder. That is the code as shipped; the
example script calls the module the same way the training script does.

## LateFusionTransformer

Encoders are `TransformerSeq` modules that already live in a shared time
axis. Fusion then:

1. Concatenates the three sequences on the feature axis
   (`64+128+1024=1216` for BERT, `64+128+512=704` for the GloVe seq
   encoders — the script actually builds `LateFusionTransformer(in_dim=1792)`
   for GloVe, matching `64+128+512` only if you recount after a different
   width; the published GloVe script uses `in_dim=1792` with seq dims
   64 / 128 / 512. 64+128+512=704. The 1792 figure is what the script
   passes; if you rebuild GloVe late-fusion from scratch, set `in_dim` to
   the real concatenated width or the `Conv1d` will not accept the input.)
2. `Conv1d` projects that width to `embed_dim=32`.
3. A 4-layer transformer over time returns the last timestep `(B, 32)`.
4. Head: `MLP(32, 32, 1)`.

On BERT-MOSEI this is the best of the six sweep methods
(MAE 0.5846, Acc-2 0.8393, F1 0.8699).

## GatedMultiTransfomerModel

Documented in [architecture.md](architecture.md#gated-multi-transformer-gmtm).
As a fusion module it consumes three `(B, T, F_i)` tensors and returns
`(B, 1)` directly. It is the only fusion whose output is already the
sentiment score.

## Shape cheat sheet (BERT, `B=4`, `T=50`)

| Module | Input | Output |
| --- | --- | --- |
| ConcatEarly | 3 × `(B,T,F_i)` | `(B, T, 877)` |
| ConcatLate | `(B,64), (B,256), (B,1024)` | `(B, 1344)` |
| TensorFusion | `(B,19), (B,39), (B,159)` | `(B, 128000)` |
| LowRankTensorFusion | `(B,32), (B,64), (B,256)` | `(B, 256)` |
| EarlyFusionTransformer | 3 × `(B,50,F_i)` or `(B,50,877)` | `(B, 32)` |
| LateFusionTransformer | 3 × `(B,50,d_i)` or `(B,50,1216)` | `(B, 32)` |
| GMTM | 3 × `(B,50,F_i)` | `(B, 1)` |

## Practical notes

- **Packed vs not.** Concat / TFN / LMF / TransformerLate are trained with
  `is_packed=True` and the `_process_1` collate (true lengths).
  TransformerEarly and GMTM use `max_pad=True` and `_process_2`.
- **Zeros vs dropping a modality.** Ablation keeps three tensors and writes
  zeros into the unused ones (`get_ablation_dataloader`). Fusion rank does
  not change.
- **Do not mix BERT widths into a GloVe checkpoint.** The `Conv1d` and LMF
  factor shapes are baked into the `.pt` file.
