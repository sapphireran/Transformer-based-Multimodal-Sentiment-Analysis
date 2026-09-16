# Personal lab notes

Working notes from reading the code and the checked-in CSVs. These are
not claims about a paper submission; they are reminders for future me
(or anyone else reading this personal repo).

## What the architecture is trying to do

GMTM is a full pairwise cross-modal transformer: every modality attends to
every other modality, the three answers are softmax-weighted, gated, then
pooled over time. That is more compute than ConcatLate or a single late
transformer on concatenated features, and it only pays off if the extra
paths carry signal.

On BERT MOSEI they mostly do not. Text-only GMTM is already 0.5687 MAE;
the full trio is 0.5640. The interesting comparison is therefore
**GMTM-text vs TransformerLate (0.5846)** — a same-task, stronger text
encoder — not "multimodal magic".

On GloVe, text is weaker, and trimodal GMTM (0.6251) pulls closer to LMF
(0.6174) while beating TransformerLate (0.6287). The MOSI GloVe transfer
table is where GMTM actually looks like a multimodal model: 0.9748 vs
everyone else ≥ 1.08.

## Zero-masking is not the same as removing a modality

Ablation loaders keep a 3-slot GMTM and write zeros into dropped streams.
The 3×3 attention grid still runs. That explains some ugly rows:

- GloVe `text+audio` is worse than `text` on both MOSEI and MOSI.
- MOSI BERT audio-only correlation is negative.
- MOSI BERT full trio is worse than text+visual.

If I ever rerun ablations, the cleaner control is a model whose
`n_modalities` actually shrinks. The current setup answers "what if this
slot is silence?" which is a useful robustness question, but it is easy
to mis-caption as "unimodal GMTM".

## MOSI numbers are not test-split numbers

`get_mosi_dataloader` concatenates train, valid, and test. The transfer
CSVs therefore mix utterances the official protocol would keep apart. Fine
for a sanity check ("does the checkpoint explode on MOSI-shaped tensors?").
Not fine for a table caption that says "MOSI test".

## Script drift vs saved CSVs

Several launch files are not in a state that would regenerate the CSVs
without edits:

1. `train_main_bert.py` uses `total_epochs=1`. The BERT main table is too
   strong to be a one-epoch run of those architectures.
2. GMTM and GloVe main training loops are commented; they only `torch.load`.
3. `train_and_test.test` does not take `dataset=` or `no_robust=`, but
   `train_GMTM_glove.py` and `train_main_glove.py` pass them.
4. `train_mosi_glove.py` imports a MultiBench module that is not vendored.
5. `train_mosi_bert.py` looks for `Concat.pt` while the sweep saves
   `ConcatLate.pt`.
6. GloVe `LateFusionTransformer(in_dim=1792)` does not match
   `64+128+512=704`. If that script is uncommented, conv weight shapes
   must match whatever was actually saved in `glove_TransformerLate.pt`.

I left the training scripts untouched in this documentation pass. The
examples import `models.py` only.

## Metric footnote

Acc7 / Acc5 here are **equal-width bins on [-3, 3]**, not the
round-to-nearest-integer protocol. Acc2 / F1 drop exact-zero gold labels
and threshold at `> 0`. When comparing to another repo's MOSI numbers,
check those two details first — they move Acc7 a lot.

## `models.py` leftovers

- `self.alpha` and the residual head path are unused.
- `trans_mems` is built and never called.
- `TransformerEncoderLayer.apply_sublayer` adds a residual of `x` onto
  itself after attention already replaced `x`. It still runs; it is just
  not a textbook Pre-LN block. I did not "fix" it because that would
  change any checkpoint that was trained with this code.
- `class Linear` is shadowed by `def Linear`.

## What I would run next (personal, not scheduled)

- CPU GMTM forward + a 20-step synthetic overfit (now covered by
  `examples/`).
- A true-subset ablation that instantiates `GatedMultiTransfomerModel(1, …)`
  and `…(2, …)` instead of zero-filling.
- A MOSI loader that keeps the official split, then re-score one BERT and
  one GloVe checkpoint.
- Replace `.cuda()` in launch scripts with the `device` already used inside
  `train()` so a laptop can smoke-test.

Until then, the CSV walkthrough in [results.md](results.md) is the record
of what was measured.
