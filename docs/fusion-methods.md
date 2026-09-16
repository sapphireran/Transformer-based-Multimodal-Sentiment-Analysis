# Fusion methods

Fusion is the only stage that sees more than one modality at once. This
repo compares six published-style baselines plus the gated transformer.

Shapes below assume a toy clip with widths `(visual, audio, text) = (8, 12, 16)`
and batch 4, time 10 — the same pack `examples/02_fusion_forward.py` prints.
BERT MOSEI widths are `(35, 74, 768)`.

## Early concatenation

`ConcatEarly` cats on the feature axis:

```text
[B, T, 8] ⊕ [B, T, 12] ⊕ [B, T, 16]  →  [B, T, 36]
```

`train_main_bert.py` then runs `LSTM(877, 1024)` with packed lengths. Early
fusion is cheap and lets a recurrent net see lip motion and words at the
same time step, but it also lets a high-variance stream (COVAREP `-inf`
replaced by zeros) pollute the shared LSTM.

GloVe early concat uses `LSTM(409, 512)` because `35+74+300 = 409`.

## Late concatenation

`ConcatLate` flattens each encoder output and cats on the batch feature
axis. Training encoders are LSTMs; the examples mean-pool time instead:

```text
[B, 8] ⊕ [B, 12] ⊕ [B, 16]  →  [B, 36]
```

BERT MOSEI late concat: LSTM hidden sizes 64 / 256 / 1024 → MLP on 1344-d.
GloVe: 64 / 256 / 512 → 832-d.

Late concat cannot model multiplicative interactions (a smile that only
matters when the words are sarcastic) unless the head MLP invents them.

## Tensor Fusion (TFN)

`TensorFusion` appends a 1 to each vector and takes the outer product
across modalities:

```text
[1 : v] ⊗ [1 : a] ⊗ [1 : t]
```

Toy widths produce `(8+1)*(12+1)*(16+1) = 1989` features. The BERT recipe
shrinks encoder outputs to 19 / 39 / 159 on purpose:

```text
20 × 40 × 160 = 128000
```

and then an MLP `128000 → 2048 → 1`. That is the memory bottleneck of the
baseline suite. `TensorFusion.forward` short-circuits to the single
tensor when only one modality is passed.

## Low-rank Tensor Fusion (LMF)

`LowRankTensorFusion` stores a factor `rank × (d_m+1) × output_dim` per
modality, multiplies the projected tensors, and mixes ranks with
`fusion_weights`. BERT MOSEI uses output 256, rank 32, encoder widths
32 / 64 / 256. The examples use output 16, rank 4, native toy widths.

LMF is the practical TFN: same multiplicative bias, far fewer parameters.
On MOSEI BERT it is the second-best baseline after late transformers
(`docs/experiments.md`).

The module builds its bias-1 column on `cuda:0` whenever a GPU is
visible, even if the rest of the model was moved elsewhere. CPU example
code is fine because `torch.cuda.is_available()` is false in this lab.

## Early transformer

`EarlyFusionTransformer` is a 1×1 `Conv1d` from the concatenated width
down to `embed_dim=32`, then a 4-layer `TransformerEncoder` with 4 heads.
The class sets `batch_first=True` on the encoder layer but then permutes
the tensor to `[T, B, 32]` before calling it. That mismatch is recorded
in `docs/known-quirks.md`. The examples lab does not instantiate this
class; it uses `TransformerFusion`, which already expects
`[B, n_modalities, d_model]`.

BERT early transformer: `n_features=877`. GloVe: `n_features=409`. The
BERT script attaches `MLP(64, 64, 1)` as the head; the GloVe script
attaches `Identity` (so the 32-d last token would need a different head
to match the checkpoint that was actually trained).

## Late transformer

Each modality goes through `TransformerSeq` (conv to a per-stream width,
4-layer encoder, **full sequence** returned). Those sequences are cat'd
on the feature axis and fed to `LateFusionTransformer`, which convs to
32-d and returns the **last time step**.

BERT widths: 64 / 128 / 1024 → late `in_dim=1216`.
GloVe widths: 64 / 128 / 512 → late `in_dim=704` in principle, but
`train_main_glove.py` passes `in_dim=1792`. See quirks.

This baseline is the strongest non-gated method on MOSEI BERT
(MAE 0.5846).

## Gated Multi-Transformer

Documented in `docs/architecture.md`. Relative to late transformers it
adds (a) pairwise cross-attention instead of independent per-stream
encoders, (b) a softmax mix over source modalities, (c) a sigmoid gate,
(d) attention pooling instead of "last time step".

Ablation loaders still pass three tensors; dropped streams are zeros.
That is why a "text-only" GMTM row still constructs `n_features=[35, 74, 768]`.

## Which collate each method needs

| method | collate | `is_packed` in `train()` |
| --- | --- | --- |
| ConcatEarly | `_process_1` | True |
| ConcatLate | `_process_1` | True |
| TensorFusion | `_process_1` | True |
| LowRankTensorFusion | `_process_1` | True |
| TransformerLate | `_process_1` | True |
| TransformerEarly | `_process_2` | False |
| GMTM | `_process_2` | False |

`examples/05_packed_vs_padded.py` prints both layouts side by side.
