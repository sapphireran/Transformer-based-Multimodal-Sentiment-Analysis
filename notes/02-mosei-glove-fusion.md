# 02 — MOSEI GloVe fusion bake-off

**Script:** `model/train_main_glove.py`
**Data:** `data/MOSEI/mosei_raw_glove.pkl`
**CSV:** `model/glove_results.csv`
**Checkpoints:** `model/checkpoints/glove_{FusionMethod}.pt`
**Text:** GloVe 840B 300-d, visual 35, audio 74, batch 32

## Why I ran this

BERT is a strong unimodal sentiment model. I wanted the same six fusion
methods with a *weaker* text front, to see whether multiplicative fusion
starts to matter when the language stream is not already solving the task.

The current script has `train()` commented out and only tests saved
checkpoints. The table below is from those weights, not from a fresh run
of the file as committed.

## Table (copied from the CSV)

| Fusion Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ------------- | ---: | ----: | ----: | ----: | ---: | --: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | **0.5287** | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

## Side-by-side with the BERT bake-off (MAE / Acc-2 / Corr)

| Method | BERT MAE | GloVe MAE | Δ MAE | BERT Acc-2 | GloVe Acc-2 | BERT Corr | GloVe Corr |
| ------ | -------: | --------: | ----: | ---------: | ----------: | --------: | ---------: |
| ConcatEarly | 0.6178 | 0.6782 | +0.0604 | 0.8084 | 0.7664 | 0.6652 | 0.5942 |
| ConcatLate | 0.6148 | 0.6329 | +0.0181 | 0.8084 | 0.7878 | 0.6637 | 0.6528 |
| LMF | 0.5970 | 0.6174 | +0.0204 | 0.8305 | 0.8107 | 0.6925 | 0.6704 |
| TFN | 0.6022 | 0.6337 | +0.0315 | 0.8244 | 0.8038 | 0.6776 | 0.6435 |
| TransformerEarly | 0.6057 | 0.6649 | +0.0592 | 0.8206 | 0.7821 | 0.6788 | 0.6063 |
| TransformerLate | 0.5846 | 0.6287 | +0.0441 | 0.8393 | 0.7975 | 0.7041 | 0.6397 |

Every method gets worse when I swap BERT for GloVe. That is the expected
direction. The **ranking** is what changed.

## Ranking flip

On BERT: `TransformerLate > LMF > TFN ≈ TransformerEarly > concat`.
On GloVe: `LMF > TransformerLate ≈ ConcatLate ≈ TFN > TransformerEarly > ConcatEarly`.

Late transformer loses its crown. LMF is first on every GloVe column.

My working explanation, not proven:

- BERT already produces a sentiment-ready 768-d sequence. A fat per-modality
  transformer can ride that. Multiplicative interactions add little.
- GloVe 300-d is lexical, not contextual. Cross-terms (LMF’s low-rank outer
  products) are a way to build “this word + this eyebrow + this pitch”
  features that BERT had already linearized.
- ConcatEarly and TransformerEarly both operate on a raw joint frame
  (`409-d`). They take the largest MAE hits in the Δ table (+0.060 / +0.059).
  Joint-time models seem more brittle when the text channels are weaker and
  the scale mismatch vs 35-d / 74-d is still there.

ConcatLate’s Δ MAE is the *smallest* (+0.018). Separate LSTMs degrade
gracefully. They just never were good enough to win.

## Implementation landmine on TransformerLate (GloVe)

`train_main_glove.py` builds

```text
TransformerSeq(35, 64), TransformerSeq(74, 128), TransformerSeq(300, 512)
LateFusionTransformer(in_dim=1792)
```

`64+128+512 = 704`, not 1792. 1792 looks like an older width (for example
`256+512+1024`). If the saved `glove_TransformerLate.pt` was trained with
matching `in_dim`, the current constructor would not load into a newly
initialized module — but `torch.load` restores the *whole* module, so eval
still works. A **retrain** from this file would Conv1d the wrong in_channels
unless I fix `in_dim=704`.

I am leaving the code alone in this docs pass. Flagged again in note 07.

## Compared to GMTM GloVe

GMTM trimodal GloVe: `MAE 0.6251`, Acc-2 `0.8079`, Corr `0.6714`.
LMF bake-off: `MAE 0.6174`, Acc-2 `0.8107`, Corr `0.6704`.

LMF is slightly better on MAE and Acc-2; Corr is a tie. So on the weak text
front, **GMTM did not beat the best classical fusion**. That is the opposite
of the BERT picture (where GMTM MAE 0.5640 beat TransformerLate 0.5846).

Hypothesis I want to test later: GMTM’s pairwise grid is capacity that
helps when text is already informative (it can route audio/vision onto a
good query) and hurts or idles when text is weak (the grid has more ways
to overfit noise). Note 03’s GloVe `text+audio` collapse is evidence in
that direction.

## Personal takeaway

Do not quote the BERT ranking as “transformers win fusion.” Quote it as
“transformers win fusion **when text is BERT**.” On GloVe, I should keep
LMF as the number to beat and treat late transformer as a second baseline,
not the default winner.

If I only have compute for one GloVe retrain, I fix `in_dim`, seed three
LMF vs GMTM vs text-only LSTM runs, and stop.
