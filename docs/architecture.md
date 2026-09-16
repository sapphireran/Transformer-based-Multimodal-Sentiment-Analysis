# Architecture

This project wraps every bake-off entry as **encoders → fusion → head**. The wrapper is `MultiFramework` in `model/train_and_test.py`. The gated model used for the ablation study is `GatedMultiTransfomerModel` in `model/models.py` (the class name keeps the original spelling).

## MultiFramework

```
inputs:  vision [B, T, 35], audio [B, T, 74], text [B, T, D_text]
           │
           ▼
      encoder_0, encoder_1, encoder_2     (Identity, LSTM, GRU, or TransformerSeq)
           │
           ▼
         fusion                           (concat / tensor / transformer / GMTM)
           │
           ▼
          head                            (MLP, LSTM+MLP, or Identity)
           │
           ▼
      scalar sentiment                    [B, 1]  (L1 / MAE training)
```

`has_padding=True` (`is_packed` in `train()`) means each encoder receives `[tensor, lengths]` so LSTM/GRU can use `pack_padded_sequence`. Transformer-early and GMTM use fixed-length padded tensors (`max_pad=True` in the dataloader) and skip packing.

`MultiFramework` also stores:

- `self.reps` — list of per-modality encoder outputs
- `self.fuseout` — fused tensor before the head

Those are handed to optional custom objectives through `objective_args_dict`. The recorded experiments use `torch.nn.L1Loss` and ignore that hook.

## Tensor layouts

Two padding modes exist in `model/data/get_dataloader.py`.

### Packed / variable length (`max_pad=False`) → `_process_1`

Returns `(modalities, lengths, ids, labels)` where each modality is `[B, T_i, F_i]` after `pad_sequence`. LSTM / GRU encoders set `has_padding=True`.

### Fixed pad (`max_pad=True`) → `_process_2`

Each clip is truncated to `max_pad_num` (default 50) and zero-padded. The batch is four tensors:

```
vision [B, 50, 35]
audio  [B, 50, 74]
text   [B, 50, 768 or 300]
label  [B, 1]
```

GMTM and transformer-early use this layout.

## Building blocks in `models.py`

| Class | Role |
| --- | --- |
| `Identity` | Pass-through encoder or head |
| `LSTM` / `GRU` / `GRUWithLinear` | Sequence encoder; optional pack + linear map |
| `MLP` | Two-layer ReLU MLP used as regression head |
| `Transformer` | Conv1d project → 4-layer encoder → last time step |
| `TransformerSeq` | Same stack, but keeps the full `[B, T, embed]` sequence |
| `TransformerEncoder` / `TransformerEncoderLayer` | Custom encoder used inside GMTM (supports cross-attn via `x_in_k`, `x_in_v`) |
| `SinusoidalPositionalEmbedding` | Fixed sin/cos positions for that custom encoder |
| `AttentionPooling` | Softmax time weights from a linear score, then weighted sum |
| `GatedMultiTransfomerModel` | Cross-modal grid + gate + pool + head |

PyTorch's `nn.TransformerEncoderLayer` is used by `Transformer`, `TransformerSeq`, `EarlyFusionTransformer`, `LateFusionTransformer`, and `TransformerFusion`. The custom `TransformerEncoder` is only used by GMTM.

## Gated multi-transformer (GMTM)

Constructor: `GatedMultiTransfomerModel(n_modalities, n_features, hyp_params)`.

Recorded BERT / GloVe runs use:

```
n_modalities = 3
n_features   = [35, 74, 768]   # BERT
             = [35, 74, 300]   # GloVe
embed_dim    = 64
num_heads    = 4
layers       = 4
attn_dropout_modalities = [0, 0, 0.1]
embed_dropout = 0.2
output_dim   = 1
```

### Forward path

1. **Project.** Each `x[i]` arrives as `[B, T, F_i]`. GMTM permutes to `[T, B, F_i]`, flattens `T*B`, runs `Linear → LayerNorm → Dropout → ReLU`, and reshapes back to `[T, B, embed_dim]`.
2. **Cross-modal grid.** For every pair `(i, j)` a `TransformerEncoder` attends from modality `i` (query) to modality `j` (key/value). That is a `3×3` stack of encoders.
3. **Weighted mix.** Softmax over a learned `modal_weights` vector (`[3]`) mixes the three views that target modality `i`.
4. **Gate.** `sigmoid(Linear(h_i)) * h_i` per modality.
5. **Concat.** Stack gated streams on the feature axis → `[B, T, embed_dim * 3]`.
6. **Attention pool.** `AttentionPooling` collapses time → `[B, 192]`.
7. **Head.** `LayerNorm → Linear(192, 96) → ReLU → Dropout → Linear(96, 1)`.

`trans_mems` (self-attention “memory” stacks) are constructed but the call is commented out in `forward`. A residual mix with `self.alpha` is also commented out. The recorded checkpoints therefore use the gated, pooled path only.

### Ablation without changing the graph

`get_ablation_dataloader` does **not** shrink `n_modalities`. Dropped streams are replaced by zeros with the official feature shape (`(50, 35)`, `(50, 74)`, `(50, 768|300)`). GMTM always sees three inputs; missing modalities contribute nothing after the projection (aside from whatever the randomly initialized projection does to exact zeros).

## Baseline encoder / head pairing

`train_main_bert.py` (MOSEI, BERT) wires the bake-off as follows.

| Fusion | Encoders | Head | Packed? |
| --- | --- | --- | --- |
| ConcatEarly | Identity × 3 | LSTM(877→1024) + MLP(1024→1) | yes |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | MLP(1344→1) | yes |
| LowRankTensorFusion | GRU+Linear → 32 / 64 / 256 | MLP(256→1) | yes |
| TensorFusion | GRU+Linear → 19 / 39 / 159 | MLP(128000→1) | yes |
| TransformerEarly | Identity × 3 | MLP(64→1) | no (fixed pad) |
| TransformerLate | TransformerSeq 35→64, 74→128, 768→1024 | MLP(32→1) | yes |

GloVe variants shrink the text encoder width (see `train_main_glove.py`). GMTM scripts use Identity encoders and an Identity head because the fusion module already includes the classifier.

## Complexity notes

`train_and_test.all_in_one_train` optionally wraps the loop with `memory_profiler.memory_usage` and prints peak RSS plus parameter count. That import is required if you call `train(..., track_complexity=True)` (the default). The synthetic toy trainer in `examples/toy_train_loop.py` skips that wrapper.

## Known implementation details (do not “fix” silently)

These are how the personal research code actually behaves:

- `EarlyFusionTransformer` builds `nn.TransformerEncoderLayer(..., batch_first=True)` but then feeds `[T, B, C]` after a permute. It also defines `self.linear` and never uses it; the module returns the last time step of the encoder, and `train_main_bert.py` applies a separate MLP head.
- `LateFusionTransformer` does the same `batch_first=True` + `[T, B, C]` mix and returns `x[-1]`.
- `TransformerEncoderLayer.apply_sublayer` applies dropout and LayerNorm on top of the already-attended tensor (the residual add is `norm(x + dropout(x))`, not `norm(x + attn(x))` relative to a stored residual). Treat this as part of the recorded model, not a drop-in fairseq clone.
- `LowRankTensorFusion` builds a ones-column on CUDA if a GPU is visible, even when the rest of the batch is on CPU. The examples pin that module to CPU and skip a CUDA device when they only need shapes.

Documenting these quirks is the point of this note: the examples exercise the real modules rather than a cleaned rewrite.
