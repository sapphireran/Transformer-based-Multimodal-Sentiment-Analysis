# Architecture

This personal repository studies **clip-level sentiment regression** from three
time-aligned streams: face, voice, and language. A clip is a short opinion
segment from CMU-MOSI or CMU-MOSEI. The label is a continuous score in
`[-3, 3]` (strongly negative → strongly positive).

Everything that trains or evaluates a model is an **encoder → fusion → head**
stack wrapped by `MultiFramework` in `model/train_and_test.py`. The gated
cross-modal transformer (`GatedMultiTransfomerModel` in `model/models.py`) is
the architecture this repo adds on top of the usual concat / tensor / early
transformer baselines.

## Tensor layout

`Affectdataset` (`model/data/get_dataloader.py`) yields one clip as:

| slot | content | typical shape |
| --- | --- | --- |
| 0 | visual Facet 4.2 | `[T, 35]` |
| 1 | acoustic COVAREP | `[T, 74]` |
| 2 | language BERT or GloVe | `[T, 768]` or `[T, 300]` |
| 3 | index (packed collate only) | scalar |
| 4 / last | label | `[1]` or `[1, 1]` |

After collate, a training step sees either:

* **packed** (`_process_1`): three padded tensors plus a length vector per
  modality, used by LSTM/GRU encoders and late concat / TFN / LMF;
* **max-pad** (`_process_2`): three dense `[B, 50, F]` tensors, used by
  `EarlyFusionTransformer` and `GatedMultiTransfomerModel`.

Time is aligned to the **first non-zero language token** when `aligned=True`
(the default). Empty-language clips are dropped by `drop_entry`.

```text
 clip
  ├─ visual [T, 35]  ──► encoder_v ──┐
  ├─ audio  [T, 74]  ──► encoder_a ──┼─► fusion ──► head ──► ŷ ∈ R
  └─ text   [T, D]   ──► encoder_t ──┘
```

`D` is 768 for BERT and 300 for GloVe. The examples lab in `examples/` builds
the same layout from Gaussian features so the fusion code can run without the
CMU pickles.

## MultiFramework

`MultiFramework` is a thin container:

1. run each encoder on its modality (optionally with pack lengths);
2. stash encoder outputs on `self.reps`;
3. call `fusion(reps)`;
4. stash the fused tensor on `self.fuseout`;
5. call `head(fused)`.

Baseline scripts (`train_main_bert.py`, `train_main_glove.py`) build a
different encoder/fusion/head triple per fusion name. GMTM scripts
(`train_GMTM_bert.py`, `train_GMTM_glove.py`) set the encoders and the head to
`Identity` because the gated module already projects, fuses, pools, and
regresses.

## Gated Multi-Transformer (GMTM)

`GatedMultiTransfomerModel` is the proposed stack. With three modalities it
does the following.

### 1. Per-modality projection

Each stream is linearly mapped from its native width to `embed_dim`
(64 in the training scripts, 8 in the CPU examples), then LayerNorm +
Dropout + ReLU. The projection is applied per time step after flattening
`[T, B, F] → [T*B, F]`.

### 2. Pairwise cross-modal transformers

For every ordered pair `(i, j)` a `TransformerEncoder` is called as

```text
h_ij = Trans_ij(query = proj_i, key = proj_j, value = proj_j)
```

That is the MulT-style pattern: modality `i` attends to modality `j`,
including `i == j` (self-attention). Each encoder has sinusoidal
positions, pre-norm sublayers, and optional future masks
(`attn_mask=True` in the training `HParams`).

### 3. Learned modality mix

The `n_modalities` pairwise outputs for a fixed query modality are stacked
and reduced with a softmax over `self.modal_weights`. The same weight
vector is shared across time and batch; it is the cheap stand-in for a
mixture-of-experts gate over source modalities.

### 4. Feature-wise sigmoid gate

`gate = σ(W h_fused)` multiplies the mixed stream elementwise. This is the
"gated" part of the name: a source that is noisy for the current clip can
shrink without dropping the whole modality from the graph.

### 5. Attention pooling over time

The three gated streams are concatenated on the feature axis to
`[B, T, embed_dim * 3]`. `AttentionPooling` scores every time step with a
linear unit and takes a weighted sum. That replaces the "last hidden
state" pooling used by the LSTM baselines.

### 6. Regression head

LayerNorm → Linear → ReLU → Dropout → Linear maps the pooled vector to a
single sentiment score. Training minimises **L1** (`nn.L1Loss`), so the
reported MAE is the training objective, not a post-hoc metric.

```text
visual ─► Linear+LN ─┐
audio  ─► Linear+LN ─┼─ pairwise Trans_ij ─ mix ─ σ-gate ─┐
text   ─► Linear+LN ─┘                                    │
                                                          ▼
                                            concat ─ attn-pool ─ MLP ─ ŷ
```

Hyper-parameters used in the BERT GMTM scripts:

| field | value |
| --- | --- |
| `embed_dim` | 64 |
| `num_heads` | 4 |
| `layers` | 4 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |

The class also stores an unused `self.alpha` residual coefficient. The
residual add is commented out in `forward`; leaving it documented here
avoids treating that parameter as part of the published recipe.

## Baseline fusions

These live in the same `models.py` file and are wired in
`train_main_bert.py` / `train_main_glove.py`.

| name | idea | BERT MOSEI encoder sketch |
| --- | --- | --- |
| ConcatEarly | cat features at every time step, then LSTM | Identity ×3 + LSTM(877) |
| ConcatLate | LSTM per modality, cat the last states | LSTM 35/74/768 → MLP(1344) |
| TensorFusion | outer product of bias-augmented vectors (TFN) | GRU → 19/39/159 → 128000-d MLP |
| LowRankTensorFusion | rank-`r` factors of the same tensor (LMF) | GRU → 32/64/256, rank 32 |
| TransformerEarly | cat features, 1×1 conv to 32-d, 4-layer encoder | Identity ×3, `n_features=877` |
| TransformerLate | `TransformerSeq` per modality, then another encoder | 64/128/1024 → 1216-d late encoder |

GloVe variants shrink the language width from 768 to 300 and retune the
hidden sizes (see `docs/fusion-methods.md`).

## Training loop

`train()` in `train_and_test.py`:

* optimiser defaults to RMSprop in the function signature; the scripts
  pass **AdamW**, `lr=1e-4`, `weight_decay=0.01`;
* gradient clip at 8;
* validation L1; the lowest-loss snapshot is saved with `torch.save(model, path)`
  (the whole `MultiFramework`, not a `state_dict`);
* optional early stop after 7 non-improving epochs;
* optional `memory_profiler` wrap when `track_complexity=True`.

`single_test()` reports MAE, MSE, Pearson, uniform Acc-7 / Acc-5, and
binary Acc-2 / F1, then draws a matplotlib confusion matrix. Headless
example code uses `model/metrics.py` instead so it never calls `plt.show()`.

## Device notes

Most training scripts call `.cuda()` at construction time. Low-rank
fusion also builds its bias-1 vector on `cuda:0` if a GPU is visible.
The examples lab forces CPU and never touches those scripts.
