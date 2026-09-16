# 08 — Reading list and personal context

Why this repo exists, what I was reading, and the boundary of the work.

## Personal context

This is a **personal** multimodal sentiment lab. MOSI / MOSEI, public
features, MIT-licensed code. No employer data, no internal models, no
production inference path.

I started from the usual stack people use for these two datasets:
CMU Multimodal SDK → aligned Facet / COVAREP / words → a MultiBench-ish
`encoders + fusion + head` loop. `models.py` still shows that family
(TFN, LMF, concat, LSTM/GRU helpers). GMTM is the one architecture I
wrote to have an opinion, not just a reproduction.

The question I actually care about is smaller than “SOTA on MOSEI”:

> If BERT text already gets Acc-2 ≈ 0.84, what is the scientific content
> of a trimodal fusion paper on this benchmark?

Notes 01–04 are me answering that for myself: the content is thin on
BERT, slightly less thin on GloVe, and audio is fragile under a MOSI
shift.

## Papers / repos that shaped the code

I am listing what the *implementation* points at, not a complete survey.

| Idea | Where it landed in this repo |
| ---- | ---------------------------- |
| CMU-MOSI (Zadeh et al.) | `data/MOSI`, opinion labels, `[-3, 3]` |
| CMU-MOSEI (Zadeh et al.) | `data/MOSEI`, main tables |
| Tensor Fusion Network (Zadeh et al. 2017) | `TensorFusion`, comment links Justin1904’s repo |
| Low-rank Multimodal Fusion (Liu et al. 2018) | `LowRankTensorFusion` |
| MulT — Multimodal Transformer (Tsai et al. 2019) | pairwise `trans[i][j]`, unused `trans_mems` |
| MultiBench / MultiZoo (Liang et al.) | `MultiFramework` shape, packed LSTM helpers, leftover `training_structures` import |
| BERT (Devlin et al.) | 768-d text pickle |
| GloVe 840B (Pennington et al.) | 300-d text pickle |
| Facet / COVAREP | 35-d / 74-d streams as shipped by the SDK recipes |

I used the SDK’s word-level align + average collapse, not a learned
aligner (no CTC, no Cross-Modal Attn from the 2018 CFA paper, no
optimal transport).

## What I deliberately did not implement

- **MISA / PMR / MMIM / Self-MM / DMD / TFR-Net / ALMT.** I have read
  several of these. Adding them before I instrument GMTM’s gates would
  just grow the CSV.
- **End-to-end raw video / wav.** I am on precomputed features. That is
  a feature-quality ceiling, not a fusion ceiling.
- **Large language model text fronts.** If I swap BERT for a frozen
  modern LM, the leftover will almost certainly shrink further. That
  would be an interesting *negative* result; it is also a different
  compute budget than this lab.
- **Emotion 6-way on MOSEI.** The pickle has extra label fields. I
  trained sentiment only.

## Benchmarks I will not pretend I ran

I have CSVs for MOSEI in-domain and MOSI merged-corpus transfer. I do
not have:

- CMU-MOSEAS, POM, IEMOCAP, MSP-Podcast, MUStARD
- Unaligned MOSEI (raw timestamps, no word collapse)
- Robustness suites (missing modality at *test* time, except the
  zero-fill ablations which are also used at train)

If I add a dataset, it gets its own note and its own split description.

## How I want to talk about this work

**Fair:** “On my MOSEI BERT protocol, a late transformer beat LMF and
concat; GMTM text-only almost matched GMTM trimodal.”

**Unfair:** “GMTM achieves Acc-7 0.48, comparable to MulT.” Different
Acc-7, different seeds, different text front, unpublished protocol.

**Fair:** “GloVe T+A failed twice (MOSEI and MOSI).”

**Unfair:** “Audio features do not contain sentiment.” They might, under
a different encoder or a different align. I showed that *these* COVAREP
streams, in *this* model, did not transfer and barely correlated
in-domain.

## Next personal experiments (ordered)

Copied from note 07 so this file can stand alone as a “what I do next”
page:

1. Three-seed GMTM text vs T+V vs T+A+V on MOSEI BERT, real slot drop.
2. Print `modal_weights` and mean gates on those runs.
3. Re-score all CSVs with rounded Acc-7.
4. Real MOSI test split, same checkpoints.
5. LMF vs GMTM vs text-only, GloVe, three seeds, fixed `in_dim`.

If I only do (1) and the leftover dies, I will write a short note that
the project succeeded as a *negative* result and stop adding fusion
blocks.

## Boundary, again

Personal repo. Public datasets. No company code. If a future clone of
this tree starts growing internal paths, that is a mistake — branch it
out of here instead of mixing it.
