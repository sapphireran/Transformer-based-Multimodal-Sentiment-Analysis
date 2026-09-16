# Fusion methods

Every method below takes a list of per-modality tensors and returns one fused tensor. Shapes assume BERT text (`D_text = 768`) unless noted. `B` is batch size, `T` is time.

## Concat early (`ConcatEarly`)

```
[B, T, 35]  ⊕_feat  [B, T, 74]  ⊕_feat  [B, T, 768]  →  [B, T, 877]
```

`torch.cat(modalities, dim=2)`. No parameters. The sequence model (LSTM in the BERT bake-off) sees a single fat feature stream from the first layer.

**When it helps.** Cheap, preserves time alignment, lets one recurrent net discover cross-modal patterns. **Cost.** One modality can drown the others; visual (35-d) is small next to BERT (768-d).

## Concat late (`ConcatLate`)

Each encoder collapses its own sequence first (LSTM last hidden, flattened). Then:

```
[B, 64] ⊕ [B, 256] ⊕ [B, 1024]  →  [B, 1344]
```

`torch.flatten` + `torch.cat(..., dim=1)`. Decision-level mix: each stream is summarized before interaction.

## Tensor fusion (`TensorFusion`)

Implements the Tensor Fusion Network outer-product trick (Zadeh et al.; original code linked in the class docstring).

For each modality `z` the module prepends a `1` (bias in homogeneous coordinates), then repeatedly does

```
m ← einsum('...i,...j->...ij', m, next)   # then flatten the last two axes
```

Three vectors `a ∈ R^{20}`, `b ∈ R^{40}`, `c ∈ R^{160}` (the BERT bake-off sizes include the extra 1) become a rank-3 tensor of size `20×40×160 = 128000`, which the MLP head then scores. Memory grows as the product of (dim+1). That is why the head is `MLP(128000, 2048, 1)`.

## Low-rank tensor fusion (`LowRankTensorFusion`)

Same multilinear idea, factored. For input dims `[d1, d2, d3]`, rank `r`, output `o`:

- factors `W_k ∈ R^{r × (d_k+1) × o}`
- each modality (plus a ones column) is mapped to `[B, r, o]`
- those three tensors are multiplied elementwise across rank slices
- a learned `fusion_weights ∈ R^{1×r}` reduces rank, plus `fusion_bias`

BERT bake-off: inputs `[32, 64, 256]`, output `256`, rank `32`. GloVe bake-off: `[32, 64, 128]`, output `128`, rank `32`.

## Transformer early (`EarlyFusionTransformer`)

1. Concatenate on the feature axis (same as concat-early) → `[B, T, 877]` (BERT) or `[B, T, 409]` (GloVe: 35+74+300).
2. `Conv1d` projects `n_features → embed_dim` (hard-coded `embed_dim = 32`).
3. A 4-layer, 4-head `TransformerEncoder` runs over time.
4. The **last** time step is returned (`x[-1]` after a `[T, B, C]` permute).

`train_main_bert.py` then applies `MLP(64, 64, 1)`. Note the width mismatch: the fusion module emits 32-d, while that MLP is built at 64-d. The GloVe script uses `Identity` as the head and lets the 32-d vector go straight to L1 (the unused `self.linear` inside the module is never called). Read the script, not just the class, before comparing numbers.

## Transformer late (`LateFusionTransformer`)

Each modality is already a sequence from `TransformerSeq` (BERT: 64 / 128 / 1024). Those sequences are concatenated on the feature axis (`in_dim=1216` BERT, `1792` GloVe: 64+128+512), projected to `embed_dim=32` with `Conv1d`, encoded, and reduced to the last step. The head is `MLP(32, 32, 1)`.

This is “late” in the sense that each stream has its own transformer first; the fusion transformer only sees already-contextualized features.

## Transformer fusion of pooled vectors (`TransformerFusion`)

Not used in the recorded bake-off scripts, but present in `models.py`. It stacks a *list of `[B, D]` vectors* as a short sequence of length `n_modalities`, runs a transformer, mean-pools, and applies a linear map. Useful if you already collapsed time.

## GMTM as a fusion module

See [architecture.md](architecture.md). In the training scripts GMTM **is** the fusion object; encoders and head are `Identity`. Interaction happens as cross-attention over time, not as an outer product of pooled vectors.

## Practical comparison (what the CSVs suggest)

On MOSEI + BERT, late transformer beat the concat and tensor baselines, and GMTM with all three modalities beat that bake-off on MAE / Acc-2 / correlation. On MOSEI + GloVe the ranking is flatter; low-rank tensor fusion is competitive with GMTM. On MOSI (models trained on MOSEI), GMTM with GloVe is the most robust transfer result in `mosi_test/`. Details in [results.md](results.md).
