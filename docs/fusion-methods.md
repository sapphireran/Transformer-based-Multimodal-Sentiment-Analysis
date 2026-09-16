# Fusion methods

All modules below live in `model/models.py`. Shapes use `B` batch, `T`
timesteps, `F` per-modality width. Loader order is always
`[vision, audio, text]`.

## 1. ConcatEarly

```python
class ConcatEarly(nn.Module):
    def forward(self, modalities):
        return torch.cat(modalities, dim=2)
```

| | |
| --- | --- |
| Input | list of `[B, T, F_i]` (Identity encoders) |
| Output | `[B, T, Σ F_i]` |
| BERT width | 35 + 74 + 768 = **877** |
| GloVe width | 35 + 74 + 300 = **409** |
| Head | `LSTM(ΣF → 1024 or 512)` + `MLP` |

Early concat assumes **aligned** time. That is true for the CMU word-aligned
pickles this repo uses. Missing modalities in the ablation loaders are
replaced with zeros of the same `[50, F]` so the cat width never changes.

## 2. ConcatLate

```python
class ConcatLate(nn.Module):
    def forward(self, modalities):
        return torch.cat([m.flatten(1) for m in modalities], dim=1)
```

| | |
| --- | --- |
| Input | list of `[B, H_i]` (LSTM last states) |
| Output | `[B, Σ H_i]` |
| BERT width | 64 + 256 + 1024 = **1344** |
| GloVe width | 64 + 256 + 512 = **832** |
| Head | `MLP(ΣH, ΣH, 1)` |

Late concat discards cross-modal structure beyond “glue the vectors”. It is
the simplest competitive baseline on MOSEI GloVe (see results note).

## 3. TensorFusion

Implementation follows
[Justin1904/TensorFusionNetworks](https://github.com/Justin1904/TensorFusionNetworks).
Each modality `z` is homogenized to `[1; z]`, then successive outer
products are taken:

```
m ← [1; vis]
m ← m ⊗ [1; aud]
m ← m ⊗ [1; txt]
```

On BERT the encoder widths are 19 / 39 / 159, so the fused vector is
`20 × 40 × 160 = 128000`. The head is `MLP(128000, 2048, 1)`.

On GloVe the text encoder is 79-d and the product is
`20 × 40 × 80 = 64000`.

`TensorFusion` has **no learned fusion parameters**. All capacity sits in
the encoders and the huge MLP.

## 4. LowRankTensorFusion

Same multilinear idea, parameterized as a rank-`r` CP-style product
([Low-rank Multimodal Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)):

```
for each modality i:
    z_i ← [1; flatten(x_i)]          # [B, F_i+1]
    factor_i : [r, F_i+1, out]       # learned
    slice_i ← z_i @ factor_i         # [B, r, out]
fused ← (slice_1 ⊙ slice_2 ⊙ …)
y ← fusion_weights @ fused + bias    # fusion_weights is [1, r]
```

BERT config in `train_main_bert.py`:

- input dims after GRUs: `[32, 64, 256]`
- `output_dim=256`, `rank=32`
- head `MLP(256, 256, 1)`

GloVe: `[32, 64, 128] → 128`, rank 32.

The module moves the homogenizing ones-vector onto CUDA when a GPU is
visible (`cuda:0`), otherwise CPU. The personal examples stay on CPU.

## 5. TransformerEarly

`EarlyFusionTransformer(n_features)`:

```
[B, T, ΣF]  --Conv1d 1×1-->  [B, 32, T]
            --permute-->     [T, B, 32]
            --4 × TransformerEncoderLayer(nhead=4)-->
            --take t = T-1--> [B, 32]
```

`n_features` is 877 (BERT) or 409 (GloVe). The BERT training script then
applies `MLP(64, 64, 1)`, which **does not match** the 32-d encoder output.
`train_main_glove.py` avoids that by using `Identity` as the head and
letting the transformer module itself be the last learned map
(`EarlyFusionTransformer` has no final `Linear` in `forward` — the
`self.linear` constructed in `__init__` is unused).

The example `02_fusion_forward.py` prints this 32-vs-64 mismatch so it is
not a surprise when you wire the BERT recipe on synthetic data.

## 6. TransformerLate

Per-modality `TransformerSeq(F_i, H_i)` keeps the **whole** sequence
`[B, T, H_i]`. Those sequences are concatenated on the last axis and fed to
`LateFusionTransformer(in_dim=Σ H_i)`, which again does 1×1 conv → 32-d
transformer → last timestep.

BERT: `64 + 128 + 1024 = 1216`.
GloVe: `64 + 128 + 512 = 704` is what a consistent recipe would use;
`train_main_glove.py` actually passes `in_dim=1792`. That only works if the
checkpoint was saved with that constructor. The synthetic example uses the
**sum of the `TransformerSeq` widths** so a randomly initialized late
fusion runs without a size error.

## 7. GMTM (gated pairwise transformers)

See [architecture.md](architecture.md). From a fusion-taxonomy point of
view GMTM is:

- **cross-modal** (every modality attends to every other),
- **gated** (sigmoid mask after the weighted mix),
- **pooled** (learned attention over time, not last-timestep).

Ablation loaders never drop a modality from the list; they zero the unused
`[50, F]` slot. GMTM therefore always sees `n=3` inputs. A zeroed stream
still has a projection and four cross-attention blocks, which is why
audio+visual (no text) does **not** match a native 2-modality model.

## Choosing a method (personal rule of thumb)

| Goal | Start with |
| --- | --- |
| Sanity check / fastest CPU demo | ConcatLate |
| Strong classical baseline | LowRankTensorFusion |
| “Does a transformer over time help?” | TransformerLate |
| Best number in the checked-in MOSEI BERT table | GMTM, all three streams |
| Diagnose whether non-text streams matter | GMTM ablation grid |

The MOSI transfer tables are much noisier (see
[experiments-and-results.md](experiments-and-results.md)); do not pick a
fusion from MOSI alone.
