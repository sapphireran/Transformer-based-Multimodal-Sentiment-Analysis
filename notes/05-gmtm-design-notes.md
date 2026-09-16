# 05 — GMTM design notes (why the code looks like this)

Personal commentary on `GatedMultiTransfomerModel` in `model/models.py`.
This is not a methods section. It is me reconstructing the intent from the
implementation and from the ablation numbers in notes 03–04.

## The name

`GatedMultiTransfomerModel` — the `o` in Transformer is missing. I am not
renaming it while full-module checkpoints depend on the class name.

GMTM was meant to be: MulT-style pairwise crossmodal transformers, plus a
**learned mix** of incoming messages, plus a **gate**, plus **attention
pooling** instead of last-step or mean pooling.

## What I borrowed

- **Pairwise cross-attn (MulT).** For each ordered pair `(i, j)`, modality
  `i` queries modality `j`. That is the `trans[i][j](proj_i, proj_j, proj_j)`
  grid. MulT also has a self-attn “memory” stack after the cross terms;
  I built `trans_mems` for that and then never called it.
- **Low-rank / TFN intuition, but not the math.** I wanted a cheap way to
  say “this modality matters more globally” (`modal_weights`) without
  forming an outer product.
- **FiLM-ish gate.** `sigmoid(Linear(h)) * h` is a per-channel, per-time,
  per-example mute. This was the part I thought would save me from audio
  noise. Note 03 says it did not, on GloVe T+A.
- **Attention pooling.** I did not want last-step (early transformer) or
  flatten-the-LSTM. A linear-softmax over time is the smallest thing that
  can pick the word where the sentiment actually happens.

## Forward pass as I intended it

```text
x_m: T × B × F_m
     → Linear+LN+Dropout+ReLU → T × B × 64          # shared time, private width
     → for each m: mix_j CrossAttn(q=m, kv=j)
     → softmax(w) · mix                               # global modality prior
     → σ(W h) ⊙ h                                     # example-wise mute
     → concat_m → B × T × 192
     → attn pool over T → B × 192
     → LN → 96 → ReLU → drop → 1
```

The residual

```text
out = α * head(z) + (1-α) * proj(z)
```

is commented out. `self.alpha` is still a parameter (init 0.2) and is
unused. I remember wanting a skip around the two-layer head when the head
was collapsing to a constant; L1 + a wide LN head did not actually need
it on BERT.

## Hyperparameters I chose on purpose vs leftovers

**On purpose**

- `embed_dim = 64`, `layers = 4`, `num_heads = 4`. Small enough to fit
  three streams at batch 32 on one GPU with `n²` attention.
- `attn_dropout_modalities = [0, 0, 0.1]`. I treated text as the stream
  that overfits. Visual=0, audio=0, text=0.1. Combined with GloVe T+A
  this may have been the wrong stream to regularize (note 03).
- `embed_dropout = 0.2` on the input projection. I wanted the Linear
  768→64 (or 300→64) to not memorize token identities.
- `output_dim = 1`, L1. Same as the bake-off, so I could compare MAE.

**Leftovers / dead**

- `modality_dropout = 0.2` — never applied. I was going to drop a whole
  modality per example (true multimodal dropout). Zero-fill ablations are
  a frozen version of that idea, not the same thing.
- `use_text_transformer = True` — never read. I was going to swap the
  text Linear for a small TransformerEncoder and did not.
- `all_steps = False` — never read. Would have been “predict per frame.”
- `attn_mask = True` turns on LayerNorm in `TransformerEncoder` but
  `create_attention_mask` returns `None`. So I paid for the flag and did
  not get causal / padding masks. Max-pad zeros are visible to attention.
- `trans_mems` constructed, unused.
- `alpha` unused.

## What the global `modal_weights` can and cannot do

`modal_weights` is shape `(3,)`. Softmax once per forward, shared across
the batch. It can learn “on this dataset, incoming audio messages get
weight 0.1.” It cannot learn “this clip is sarcastic, up-weight audio.”

The per-example capacity is only the sigmoid gate, and that gate sees
`h_fused` *after* the global mix. If the mix already drowned text in
audio, the gate can only shrink the drowned vector, not un-drown it.

That ordering — mix first, gate second — is the main design regret I
would revisit before a retrain. Gate-then-mix, or per-example weights
`softmax(Linear(h_j))`, is the obvious next architecture tweak.

## Why Identity encoders

I did not put LSTMs in front of GMTM. The pairwise transformers *are*
the temporal model. Adding LSTM + GMTM would have doubled the “who
owns time” question and made ablations messier.

Cost: GMTM has to learn temporal structure from scratch in 4 layers at
width 64. Benefit: the fusion bake-off already covers “LSTM then fuse.”

## Why I still used `n=3` for unimodal rows

Laziness and checkpoint compatibility. One constructor, one pickle
layout, zero-fill the rest. The scientific cost is documented in note 03:
audio-only is not a unimodal audio network. If I want a real unimodal
number I need `GatedMultiTransfomerModel(1, [74], ...)`.

## Interaction with padding

Max-pad 50, no padding mask. Short MOSI clips (many are much shorter
than 50) attend over a sea of zeros. Attention pooling can learn to
avoid them; pairwise cross-attn can also *invent* structure from the
zero tail. This is a plausible contributor to the MOSI audio-only
negative correlations (note 04): a zero tail plus a global audio prior
is a stable garbage pattern.

## Capacity sketch (order of magnitude)

For `n=3`, `embed_dim=64`, `layers=4`, `heads=4`:

- 9 crossmodal encoders + 3 unused mem encoders
- Each encoder is 4 layers of MHA + 4× FF
- Plus 3 small input MLPs, 3 gates, 1 pool, 1 head

I never wrote the exact param count into a CSV. `all_in_one_train`
prints it. From memory this was on the order of a few million, dominated
by the 9 encoders. The unused `trans_mems` are real wasted parameters
in the optimizer unless I exclude them — and I did not exclude them.
They get grad=0 if unused… actually unused modules still sit in
`model.parameters()`. If nothing touches them they still get AdamW
weight decay toward 0. Harmless except for the memory.

## What I would keep vs drop in a v2

**Keep:** pairwise cross-attn, attention pooling, L1, the 64/4/4 shape,
Identity encoders.

**Drop or finish:** `trans_mems`, `alpha`, dead HParams, global-only
`modal_weights`.

**Add:** padding mask, per-example mix, true modality dropout during
train, `n_modalities` that matches the subset, `state_dict` checkpoints.

## Personal takeaway

GMTM is a MulT grid with two extra knobs (global mix, gate) that I did
not instrument. The BERT leftover of 0.005 MAE is too small to credit
those knobs. The GloVe T+A collapse is large enough to blame them. v2
should log the knobs or lose them.
