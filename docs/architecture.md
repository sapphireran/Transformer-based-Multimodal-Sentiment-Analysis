# Architecture notes

This is the design as implemented in `model/models.py` and wired in the train scripts.
I am describing the code, not a paper-ready method section.

## Shared wrapper

`train_and_test.MultiFramework` is always:

```text
encoders[i](modality_i)  →  fusion(list_of_reps)  →  head(fused)
```

Packed mode (`is_packed=True`) feeds `[inputs, lengths]` into each encoder. That is the
LSTM / GRU path (`has_padding=True` on those modules). TransformerEarly and GMTM use
`max_pad=True` dataloaders and `is_packed=False`.

Objective for every logged run: `nn.L1Loss` on a scalar. Optimizer in the scripts:
AdamW, `lr=1e-4`, `weight_decay=0.01`, grad clip 8.

## Feature widths

Order is always **visual, audio, text**.

| Front | Visual | Audio | Text | Concat width |
| ----- | ------ | ----- | ---- | ------------ |
| BERT  | 35 | 74 | 768 | 877 |
| GloVe | 35 | 74 | 300 | 409 |

## Fusion bake-off (the six methods)

### ConcatEarly

Identity encoders. Concatenate on the feature axis (`dim=2`), then an LSTM over the
joint sequence and an MLP to 1.

- BERT: `LSTM(877 → 1024) + MLP(1024 → 1)`
- GloVe: `LSTM(409 → 512) + MLP(512 → 1)`

This is the “just smash the channels” baseline. It has to learn a shared recurrent
dynamics for three very different feature scales.

### ConcatLate

Independent LSTMs, flatten last hidden, concatenate, MLP.

- BERT: `64 + 256 + 1024 = 1344` → MLP
- GloVe: `64 + 256 + 512 = 832` → MLP

Each modality gets its own temporal model. Fusion never sees time, only summaries.

### TensorFusion (TFN)

GRU+linear encoders to 19 / 39 / 159 (BERT) or 19 / 39 / 79 (GloVe), then the outer
product with a leading 1 appended per modality. Head is huge: `MLP(128000 → 2048 → 1)`
on BERT, `MLP(64000 → 2048 → 1)` on GloVe.

This is the original Zadeh et al. construction. Memory-heavy. I kept the output ranks
odd (`19, 39, 159`) so `(d+1)` multiplies cleanly toward those flattened sizes.

### LowRankTensorFusion (LMF)

Same GRU+linear idea, smaller factors, rank 32.

- BERT factors: 32, 64, 256 → output 256
- GloVe factors: 32, 64, 128 → output 128

On MOSEI GloVe this was the **best** of the six. On MOSEI BERT it was second to
TransformerLate. See [`../notes/02-mosei-glove-fusion.md`](../notes/02-mosei-glove-fusion.md).

### TransformerEarly

Identity encoders, concat on feature dim, `Conv1d` to `embed_dim=32`, 4-layer
`TransformerEncoder` (4 heads), take the **last time step**. BERT then uses
`MLP(64 → 1)` — that 64 vs 32 mismatch is real in the current file. GloVe uses
`Identity` as the head and the fusion module already ends at 32-d.

`EarlyFusionTransformer.forward` permutes to `(S, B, D)` even though the encoder layer
is constructed with `batch_first=True`. I have not isolated how much that scramble
costs. Flagged in [`../notes/07-failure-modes-and-open-questions.md`](../notes/07-failure-modes-and-open-questions.md).

### TransformerLate

Per-modality `TransformerSeq` (Conv1d + 4-layer encoder, **full sequence** out), concat
on feature dim, then `LateFusionTransformer` (Conv1d to 32, 4 layers, last step) + MLP.

- BERT encoder widths: 64 / 128 / 1024, fused in_dim 1216
- GloVe encoder widths: 64 / 128 / 512, fused in_dim 704? **Wait — the script says `in_dim=1792`.**

Rechecking `train_main_glove.py`: `LateFusionTransformer(in_dim=1792)` with encoders
64+128+512 = 704. That 1792 is leftover from an older width set (maybe 256+512+1024).
If I retrain GloVe TransformerLate, this will either crash or silently Conv1d the
wrong channel count. Another open item.

BERT late transformer was the bake-off winner on MOSEI.

## GMTM (`GatedMultiTransfomerModel`)

This is the model I actually care about. Forward pass, as written:

1. **Project** each modality with `Linear → LayerNorm → Dropout → ReLU` into
   `embed_dim` (64 in the HParams used for training).
2. **Pairwise crossmodal transformers.** For every ordered pair `(i, j)`,
   `trans[i][j](q=proj_i, k=proj_j, v=proj_j)`. That is a full `n²` grid.
3. **Softmax modality weights** (`self.modal_weights`, length `n_modalities`) mix the
   `n` incoming messages for modality `i`.
4. **Gate:** `sigmoid(Linear(h)) * h` per modality.
5. Concatenate the `n` gated sequences on the feature axis → `B × T × (n * embed_dim)`.
6. **AttentionPooling** over time (linear → softmax → weighted sum).
7. **Classification head:** LayerNorm → Linear half → ReLU → Dropout → Linear to 1.

`trans_mems` (self-attention “memory” stacks) are constructed and **never called**.
A residual path through `self.alpha` is constructed / commented and also unused.

HParams I used for both BERT and GloVe GMTM runs:

```text
num_heads = 4
layers = 4
embed_dim = 64
attn_dropout = 0.1
attn_dropout_modalities = [0, 0, 0.1]   # extra dropout on the last source (text)
relu_dropout = 0.1
res_dropout = 0.1
out_dropout = 0.1
embed_dropout = 0.2
attn_mask = True                         # layer_norm on, but mask factory returns None
output_dim = 1
modality_dropout = 0.2                   # unused
use_text_transformer = True              # unused
```

Ablations do **not** shrink `n_modalities`. Missing modalities are zero tensors of the
right shape (see [`data-pipeline.md`](data-pipeline.md)). The pairwise grid still runs.
That is a weaker ablation than “build a 1- or 2-modality GMTM,” and it matters when I
read the audio-only / vision-only rows.

## Encoders used as parts

| Module | Behavior |
| ------ | -------- |
| `Identity` | Pass-through. Used whenever fusion wants raw sequences |
| `LSTM` / `GRU` / `GRUWithLinear` | Packed last hidden (or last layer) |
| `Transformer` | Conv1d + encoder, **last** time step only |
| `TransformerSeq` | Same, but returns the full `B × T × D` sequence |
| `MLP` | Two-layer ReLU MLP. `output_each_layer` exists for older MultiBench losses |
| `AttentionPooling` | Time-wise attention, used only by GMTM |

## What I would change before a clean retrain

Not doing these in this docs pass. Just so I do not forget:

1. Save `state_dict` + a small config json, not the full module.
2. Fix `batch_first` vs permute in the early/late transformer blocks.
3. Confirm GloVe `LateFusionTransformer(in_dim=...)`.
4. Either use `trans_mems` / `alpha` or delete them.
5. Make GMTM ablations change `n_modalities` instead of zero-filling.
6. Stop opening a GUI confusion matrix inside `single_test`.
