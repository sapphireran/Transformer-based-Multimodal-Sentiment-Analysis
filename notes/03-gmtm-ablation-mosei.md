# 03 — GMTM modality ablations on MOSEI

**Scripts:** `model/train_GMTM_bert.py`, `model/train_GMTM_glove.py`
**Data:** `mosei_raw_bert.pkl`, `mosei_raw_glove.pkl`
**CSVs:** `model/ablation_results.csv`, `model/ablation_glove_results.csv`
**Checkpoints:** `model/checkpoints/ablation/model_{mods}.pt` and `model_glove_{mods}.pt`
**Model:** `GatedMultiTransfomerModel`, Identity encoders, max-pad 50, L1, AdamW

## Why I ran this

The bake-off tells me which fusion block wins when all three modalities are
present. It does not tell me whether audio and vision are *used*. GMTM was
supposed to be the model that could learn to gate them. The 7-way subset
grid is the test of that story.

Important method note: subsets are implemented by **zero-filling** dropped
modalities, not by building a smaller GMTM. The 3×3 crossmodal grid always
runs. See `docs/data-pipeline.md` and `docs/architecture.md`.

`train_GMTM_bert.py` currently only lists the full triple; the other six
combos are commented. The BERT CSV still has all seven rows, so those
weights / runs happened from an earlier version of the loop.

## BERT table

| Modalities | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ---------- | ---: | ----: | ----: | ----: | ---: | --: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

## GloVe table

| Modalities | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ---------- | ---: | ----: | ----: | ----: | ---: | --: |
| text | 0.6616 | 0.4220 | 0.5091 | **0.8118** | 0.6557 | **0.8509** |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

## Reading the BERT grid

**Text-only is a complete system.** MAE 0.5687 is already better than every
method in the BERT fusion bake-off except GMTM itself. Acc-2 0.8404 is
within 0.002 of the full model. If I had only shipped text-only GMTM, I
would have almost the same polarity classifier.

**Audio-only and visual-only are not sentiment models.** Corr 0.11 and 0.21.
Acc-2 ~0.63 is the kind of number you get when the model leans positive on
a positive-skewed set. F1 looks less embarrassing (0.75–0.77) because a
majority-ish positive predictor can still get a decent binary F1. I should
not cite F1 alone for these rows.

**A+V without text ≈ visual-only.** MAE 0.8220 vs visual 0.8217, Corr 0.2231
vs 0.2061. Audio does not rescue vision. There is no hidden multimodal
sentiment channel that appears only when the two non-text streams meet.

**T+A and T+V are essentially text.** T+A is a hair better than text on MAE
and Acc-7; T+V is a hair better on Acc-7 and a hair *worse* on Acc-2 and
Corr. The trimodal row is the best MAE / Acc-2 / Corr / F1, but the deltas
vs text-only are:

```text
MAE   0.5687 → 0.5640   (−0.0047)
Acc-2 0.8404 → 0.8429   (+0.0025)
Corr  0.7202 → 0.7255   (+0.0053)
F1    0.8741 → 0.8777   (+0.0036)
```

That is the “leftover” I mentioned in the README. It is real (same direction
on four metrics) and small. I would not bet it survives a three-seed test
without checking.

Acc-7 is the one column where T+V (0.4838) beats trimodal (0.4827). Noise.

## Reading the GloVe grid

**Text-only is weaker, as expected.** MAE 0.6616 vs BERT’s 0.5687. Acc-2
0.8118 vs 0.8404. This is the same story as note 02.

**Trimodal GloVe is the first time extra modalities clearly help GMTM.**
MAE drops from 0.6616 to 0.6251 (−0.0365). Corr rises 0.6557 → 0.6714.
Acc-7 0.4220 → 0.4467. Acc-2 does *not* improve (0.8118 → 0.8079) and F1
dips (0.8509 → 0.8474). So: better regression, not better polarity. That
is consistent with audio/vision adding intensity / degree, not sign.

**text+audio is a failure row.** MAE 0.7200, Acc-7 0.3503, Corr 0.5503 —
all worse than text-only, and Acc-7 is the worst number in the entire GloVe
grid including audio-only. Something about pairing GloVe text with audio
*inside this GMTM / zero-fill setup* is actively harmful.

I do not have training curves, so I cannot say overfit vs optimization.
Candidates I want to check:

1. Zero-filled vision is not a neutral absence. Cross-attn from text to a
   zero stream, plus audio, may inject a systematic bias.
2. `attn_dropout_modalities = [0, 0, 0.1]` treats the last slot (text) as
   the one that needs dropout. Combined with a live audio stream, text
   might get regularized just enough to lose to audio noise.
3. Single seed. Could be a dead run. I should not theorize past one repeat.

**text+visual ≈ text-only** on MAE (0.6612 vs 0.6616) and a bit worse on
Acc-2. Vision is at least *not* toxic in the way audio was.

**A+V without text** again tracks visual-only. Corr 0.2345 is the high
water mark for non-text, still useless next to any text row.

## BERT vs GloVe, same GMTM, same subsets

| Subset | BERT MAE | GloVe MAE | BERT Corr | GloVe Corr |
| ------ | -------: | --------: | --------: | ---------: |
| T | 0.5687 | 0.6616 | 0.7202 | 0.6557 |
| A | 0.8306 | 0.8232 | 0.1124 | 0.2067 |
| V | 0.8217 | 0.8139 | 0.2061 | 0.2256 |
| T+A | 0.5659 | 0.7200 | 0.7227 | 0.5503 |
| T+V | 0.5667 | 0.6612 | 0.7143 | 0.6459 |
| A+V | 0.8220 | 0.8141 | 0.2231 | 0.2345 |
| T+A+V | 0.5640 | 0.6251 | 0.7255 | 0.6714 |

Audio-only / visual-only are *slightly better* under the GloVe script than
under BERT. That sounds backwards until I remember the model still has a
text slot filled with zeros — the rest of the net is not identical because
the text Linear is 300 vs 768, and the runs are independent. I would not
read a deep story into A/V unimodal deltas of ~0.01 MAE.

The rows that matter are T, T+A, T+A+V.

## Did the gates work?

If gating did its job, T+A and T+V would never be *worse* than T, and A / V
would be ignored when they are noise. On BERT that mostly happened (no
harmful pairs, tiny trimodal gain). On GloVe it did **not** happen for
audio.

So: the gate is not a guarantee. It is a hope that trained on BERT-scale
text and did not transfer to the GloVe pairing.

`self.modal_weights` is a global softmax, not per-example. A clip that
needs audio and a clip that needs audio to shut up share the same three
scalars. That is a design limit, not just a training accident. Per-example
gates exist (`gating_linears`) but they sit *after* the weighted mix.

## Personal takeaway

- Quote BERT GMTM as “text-only already solves polarity; trimodal nicks MAE.”
- Quote GloVe GMTM as “trimodal helps regression; T+A is a red row; do not
  average it away.”
- Next ablation should (a) drop slots for real, (b) three seeds, (c) log
  `softmax(modal_weights)` and mean gate activations per subset.
- I should add a frozen-text + trained-audio-adapter run. If audio cannot
  help a frozen BERT/GloVe text tower, I will stop pretending the leftover
  is a fusion research problem and treat it as a feature-quality problem.
