# 04 — MOSEI → MOSI transfer

**Scripts:** `model/mosi_test/train_mosi_bert.py`, `train_mosi_glove.py`,
`mult_bert_mosi.py`, `mult_glove_mosi.py`
**Data:** `data/MOSI/mosi_raw_bert.pkl`, `mosi_raw_glove.pkl`
**CSVs:** the four files in `model/mosi_test/`
**Weights:** MOSEI checkpoints. Nothing in this folder trains on MOSI.

## Why I ran this

MOSI is the smaller, older sibling of MOSEI (YouTube movie reviews, same
Facet / COVAREP / word-aligned recipe, same `[-3, 3]` label). I wanted to
know whether a MOSEI-trained fusion head still ranks methods the same way
on a related corpus, or whether I had overfit MOSEI’s talk-show / vlog mix.

## The evaluation is not a MOSI test set

`get_mosi_dataloader` concatenates MOSI train + valid + test and scores the
lot. So does the ablation variant. Every number below is **in-corpus MOSI
including clips that standard splits call train**.

I did it as a sanity check: “do these weights produce plausible sentiment
on MOSI-looking tensors?” I did **not** do it as a number I would put next
to Tsai et al. or Yu et al.

If any future note quotes these CSVs, it has to repeat this paragraph.

Also: `train_mosi_glove.py` still imports
`training_structures.Supervised_Learning`. The CSV exists, so that run
happened in an environment where MultiBench was on `PYTHONPATH`, or I
swapped the import after. Re-running the file as committed may fail.

## Fusion transfer, BERT (MOSEI ckpts → merged MOSI)

From `mosi_bert_results.csv`. Late-concat is named `Concat` here.

| Fusion Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ------------- | ---: | ----: | ----: | ----: | ---: | --: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

The GMTM row is copied from the ablation CSV’s trimodal line (same 0.9493).
The fusion script itself does not train GMTM; I pasted it into this table
so I could look at one ranking.

## Fusion transfer, GloVe

From `mosi_glove_results.csv` plus GMTM trimodal from
`ablation_mosi_glove_results.csv`.

| Fusion Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ------------- | ---: | ----: | ----: | ----: | ---: | --: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GatedMultiTransfomer | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

## What survives the domain shift

**Absolute numbers get worse.** MOSEI BERT TransformerLate MAE 0.5846 →
MOSI 0.8986. That is a large jump even allowing for the merged split and
MOSI being a harder / smaller-style distribution. I am not going to
interpret the delta as a precise generalization gap until I re-score a
real MOSI test split.

**BERT ranking: late transformer still wins the six.** Same champion as
in-domain MOSEI. LMF is no longer a clear second (MAE 0.9525, similar to
early concat / early transformer). Late concat *collapses* (MAE 1.0056,
worst Acc-2 among the six except it is 0.7339 — actually TFN Acc-2 is
fine at 0.7705 with bad MAE). The late-concat LSTM summaries look more
MOSEI-specific than the late transformer sequence model.

**GloVe ranking: GMTM wins, LMF does not transfer its MOSEI crown.**
In-domain GloVe LMF was 0.6174; on merged MOSI it is 1.0825, statistically
tied with Concat and TransformerLate, while GMTM is 0.9748. That is the
one place GMTM looks like it has a robustness story vs LMF.

I can think of two boring explanations I have to kill before I like the
interesting one:

1. GMTM was trained longer / differently (note 06: GMTM comments say 20–50
   epochs; the fusion GloVe train call is commented and I do not remember
   the epoch count).
2. Merged-split scoring rewards models that sit near the MOSI label prior.
   GMTM’s attention pool might just be better calibrated.

Interesting explanation, if 1 and 2 die: pairwise crossmodal routing
overfits MOSEI *less* than LMF’s static factors when the text front is
GloVe and the talkers change.

**GMTM does *not* win the BERT transfer table.** 0.9493 MAE vs late
transformer 0.8986. So the robustness story is GloVe-specific in this
data, not a general GMTM property.

## GMTM subset transfer, BERT

From `ablation_mosi_results.csv`:

| Modalities | MAE | Acc-7 | Acc-2 | Corr | F1 |
| ---------- | ---: | ----: | ----: | ---: | --: |
| text | 0.9363 | 0.3170 | 0.7667 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.1681 | 0.4509 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.1663 | 0.5165 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.3055 | 0.7786 | 0.6494 | 0.7715 |
| text+visual | **0.9044** | **0.3257** | **0.7796** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.1654 | 0.5208 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.3078 | 0.7638 | 0.6403 | 0.7626 |

Audio-only Corr is **negative**. Vision-only is ~0. The non-text streams
did not transfer at all; they anti-transferred.

**T+V is the best BERT GMTM row on MOSI**, better than trimodal and better
than text-only. Adding audio to T+V (the trimodal row) *hurts* (0.9044 →
0.9493 MAE, Corr 0.6623 → 0.6403). That is the cleanest evidence I have
that audio is a liability under shift, even with BERT.

T+A ≈ text (0.9349 vs 0.9363). Audio is ~neutral when vision is absent and
harmful when vision is present. I do not have a mechanism. Could just be
one seed.

## GMTM subset transfer, GloVe

From `ablation_mosi_glove_results.csv`:

| Modalities | MAE | Acc-7 | Acc-2 | Corr | F1 |
| ---------- | ---: | ----: | ----: | ---: | --: |
| text | 1.0085 | 0.3014 | 0.7317 | 0.5959 | 0.7343 |
| audio | 1.3799 | 0.1681 | 0.4696 | −0.0720 | 0.5513 |
| visual | 1.3692 | 0.1667 | 0.5036 | 0.0409 | 0.6177 |
| text+audio | 1.2863 | 0.1745 | 0.6033 | 0.3524 | 0.5141 |
| text+visual | 1.0057 | 0.2739 | 0.7235 | 0.5967 | 0.7446 |
| audio+visual | 1.3734 | 0.1672 | 0.4998 | −0.0042 | 0.6024 |
| text+audio+visual | **0.9748** | **0.3152** | **0.7355** | **0.6177** | 0.7381 |

GloVe T+A is again a disaster (MAE 1.2863, F1 0.5141), same family of
failure as MOSEI GloVe T+A. Trimodal is the best GloVe MOSI row — so on
this front, audio is only usable in the company of vision, or the trimodal
checkpoint just landed better.

I should not average the BERT and GloVe audio stories. They disagree.

## What I will not conclude

- I will not say “GMTM generalizes to MOSI better than TransformerLate.”
  That is false on BERT and only true on GloVe, under a merged split.
- I will not say “audio never transfers.” I will say “audio-only never
  transferred; audio as a third stream is harmful on BERT MOSI and on
  GloVe T+A.”
- I will not put these Acc-7 numbers next to a paper table.

## Next run, if I touch MOSI again

1. Use `get_dataloader(..., data_type='mosi')` so test is test.
2. Fix the MultiBench import in `train_mosi_glove.py`.
3. Score the **same** MOSEI checkpoints on a real MOSI test split; keep
   this merged-split CSV as a historical appendix.
4. Train a MOSI-from-scratch GMTM text-only vs trimodal, three seeds.
   Transfer and in-domain are different questions.

## Personal takeaway

MOSEI success does not give me MOSI audio. If I care about robustness, the
first experiment is **drop audio** (BERT T+V beat T+A+V here) and **fix
the split**. Everything else is commentary.
