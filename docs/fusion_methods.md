# Fusion methods

This page is the wiring diagram for the six `train_main_*.py` recipes plus
GMTM. Shapes assume a **BERT-text MOSEI** batch unless a GloVe note says
otherwise.

Notation:

- `V, A, T` — visual / audio / text sequences
- `enc_*` — unimodal encoder output
- `B, t, F` — batch, time, feature

## Shared wrapper

```python
model = MultiFramework(encoders, fusion, head, has_padding=is_packed)
```

`is_packed` is `False` only for `TransformerEarly`. Everything else uses the
packed collate (`_process_1`) so LSTMs/GRUs can call `pack_padded_sequence`.

Checkpoints are saved as a **whole `MultiFramework`** via `torch.save(model,
path)`, not as a `state_dict`. Load with `torch.load` under the same
`models.py` / `train_and_test.py` class names.

## ConcatEarly

**Idea.** Merge raw feature axes first; let one sequence model see all
channels.

| Piece | BERT MOSEI | GloVe MOSEI |
| --- | --- | --- |
| Encoders | 3× `Identity` | 3× `Identity` |
| Fusion | `ConcatEarly` → `(B, t, 877)` | `(B, t, 409)` |
| Head | `LSTM(877, 1024, has_padding=True)` + `MLP(1024, 1024, 1)` | `LSTM(409, 512)` + `MLP(512, 512, 1)` |
| Packed | yes | yes |

`ConcatEarly.forward` is `torch.cat(modalities, dim=2)`. Time must already be
aligned.

**When it helps.** Cheap shared temporal model. **When it hurts.** Visual /
audio scales can drown text unless you normalize (the loader’s optional
`z_norm`).

## ConcatLate

**Idea.** Encode each stream, flatten, concatenate.

| Piece | BERT MOSEI | GloVe MOSEI |
| --- | --- | --- |
| Visual encoder | `LSTM(35, 64)` | `LSTM(35, 64)` |
| Audio encoder | `LSTM(74, 256)` | `LSTM(74, 256)` |
| Text encoder | `LSTM(768, 1024)` | `LSTM(300, 512)` |
| Fusion | `ConcatLate` → `64+256+1024=1344` | `64+256+512=832` |
| Head | `MLP(1344, 1344, 1)` | `MLP(832, 832, 1)` |

`LSTM` with `has_padding=True` returns the last hidden state, then
`ConcatLate` flattens (already rank-2) and `cat`s.

This is the “late fusion” baseline most multimodal papers start from.

## TensorFusion (TFN)

**Idea.** Model multiplicative interactions: each fused coordinate is a
product of (augmented) unimodal coordinates.

Encoders (`GRUWithLinear`, packed):

| Stream | BERT | GloVe |
| --- | --- | --- |
| Visual | `35 → 64 → 19` | `35 → 64 → 19` |
| Audio | `74 → 256 → 39` | `74 → 256 → 39` |
| Text | `768 → 1024 → 159` | `300 → 512 → 79` |

Each vector is prepended with a `1`, then successive Einstein products build
a tensor of size `(19+1)*(39+1)*(159+1) = 128000` (BERT) or
`20*40*80 = 64000` (GloVe). The head is `MLP(128000, 2048, 1)` or
`MLP(64000, 2048, 1)`.

This is faithful to [Tensor Fusion Networks](https://github.com/Justin1904/TensorFusionNetworks)
and is the most memory-hungry recipe in the sweep.

## LowRankTensorFusion (LMF)

**Idea.** Same multiplicative story as TFN, but each modality has a factor
`[rank, F_i+1, out_dim]` and the rank slices are multiplied then contracted.

BERT recipe:

- Encoders: `GRUWithLinear` → `32`, `64`, `256`
- Fusion: `LowRankTensorFusion([32, 64, 256], output_dim=256, rank=32)`
- Head: `MLP(256, 256, 1)`

GloVe recipe uses output widths `32, 64, 128` and `MLP(128, 128, 1)`.

LMF is the usual practical substitute for TFN. In the recorded BERT MOSEI
table it beats TFN on MAE (`0.5970` vs `0.6022`) and Acc2.

## TransformerEarly

**Idea.** Concatenate raw features, project to a small token size, run a
shared transformer over time.

- Encoders: identity
- Fusion: `EarlyFusionTransformer(n_features=877)` (GloVe: `409`)
- Head in `train_main_bert.py`: `MLP(64, 64, 1)`
- Head in `train_main_glove.py`: `Identity` (the fusion module already ends
  in a linear to 1 inside some revisions; the class as checked in returns the
  last hidden of size `embed_dim=32` and does **not** apply `self.linear`)
- `max_pad=True`, `is_packed=False`

The fusion class allocates `self.linear = nn.Linear(32, 1)` but `forward`
returns the encoder state, not `self.linear(...)`. Pair the module with a
head that matches the **32-D** last timestep unless you change `forward`.

## TransformerLate

**Idea.** Per-modality transformers that keep time, then one fusion
transformer over the concatenated feature axis.

BERT encoders:

```text
TransformerSeq(35, 64)
TransformerSeq(74, 128)
TransformerSeq(768, 1024)
```

Fusion: `LateFusionTransformer(in_dim=1216)` because `64+128+1024=1216`.
Head: `MLP(32, 32, 1)` on the last fusion timestep.

GloVe encoders use text width 512; concatenated encoder width is `704`. The
checked-in script still constructs `LateFusionTransformer(in_dim=1792)`. If
you retrain, set `in_dim` to the real concat width or the `Conv1d` will
mismatch.

This recipe is the **best fusion-sweep row** on MOSEI BERT
(`MAE 0.5846`, `Corr 0.7041`, `Acc2 0.8393`).

## Gated Multi-Transformer (GMTM)

Not part of `fusion_methods = [...]` in `train_main_*.py`. It is trained from
`train_GMTM_*.py`:

```python
encoders = [Identity, Identity, Identity]
fusion   = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=HParams)
head     = Identity
is_packed = False          # max-padded batches
```

Pairwise cross-modal transformers + softmax modality weights + sigmoid gates
+ attention pooling. Full description: [architecture.md](architecture.md).

Ablation variants keep the same 3-stream module and **zero** unused
modalities rather than rebuilding a 1- or 2-stream network. That keeps
checkpoint shapes comparable and isolates “does this stream carry signal?”
from “did we change capacity?”.

## Packed vs padded: which methods need which

| Method | Collate | Why |
| --- | --- | --- |
| ConcatEarly / Late, TFN, LMF | `_process_1` (packed) | LSTM/GRU `has_padding=True` |
| TransformerLate | packed in the main scripts | `TransformerSeq` does not read lengths; packing still changes batch time |
| TransformerEarly, GMTM | `_process_2` (max pad 50) | modules index a fixed `T` and do not take length tuples |

`examples/packed_vs_padded.py` prints both layouts from the same synthetic
utterances.

## Choosing a method for a new personal experiment

1. **Debug the data** with ConcatLate — fewest moving parts, readable widths.
2. **Need interactions** without TFN’s 128k-D vector — use LMF.
3. **Need temporal cross-talk after unimodal context** — TransformerLate.
4. **Need leave-one-modality-out with a single architecture** — GMTM.
5. Avoid TFN on small GPUs; prefer LMF or GMTM.

The synthetic walkthrough `examples/fusion_walkthrough.py` constructs each
fusion module on CPU and prints output shapes so you can sanity-check a
change before touching real pickles.
