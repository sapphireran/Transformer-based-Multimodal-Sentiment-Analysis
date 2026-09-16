# Next sweep playbook

Personal, not automated. I will not run this in the current docs-only
pass. This is the sheet I want taped next to the GPU when I come back.

Related: [`../notes/06-hyperparams-and-training-protocol.md`](../notes/06-hyperparams-and-training-protocol.md),
[`../notes/07-failure-modes-and-open-questions.md`](../notes/07-failure-modes-and-open-questions.md).

## Sweep 1 — Is the BERT leftover real?

**Model:** GMTM, BERT pickle, max-pad 50.
**Change vs old runs:** `n_modalities` matches the subset. No zero-fill.
**Cells:** `{text}`, `{text,visual}`, `{text,audio,visual}`.
**Seeds:** 13, 37, 101.
**Train:** 40 epochs, patience 8, AdamW 1e-4, wd 0.01, L1, batch 32.
**Log per epoch:** train L1, valid L1, valid Acc-2.
**Log once at best valid:** `softmax(modal_weights)`, mean `|gate|` per
modality on valid.
**Success:** leftover MAE (TAV − T) has mean < 0.01 and CI crossing 0.
Then I write a negative-result addendum to note 03 and stop adding
fusion blocks.
**Failure (leftover ≥ 0.015 mean):** instrument gates, try per-example
mix (note 05), do not add a fourth architecture.

## Sweep 2 — GloVe: LMF vs GMTM vs text-only

**Fix first:** `LateFusionTransformer(in_dim=704)` if I include late
transformer; not required for this trio.
**Cells:** LMF (existing widths), GMTM trimodal, GMTM text-only.
**Seeds / opt:** same as sweep 1, 20–40 epochs (GloVe fitted faster).
**Success:** LMF still wins MAE on the mean. Then GMTM is a BERT-only
story and I stop selling it as a general fusion.
**Failure (GMTM wins):** only then do I spend time on GloVe T+A.

## Sweep 3 — T+A autopsy (only if I care after 1–2)

One seed is enough for a diagnostic, three if I want a claim.

- Train GMTM T+A with `attn_dropout_modalities = [0, 0, 0]` vs `[0, 0.1, 0]`.
- Print valid MAE every epoch against a text-only control on the same seed.
- If T+A still loses by ~0.05 MAE, look at alignment / COVAREP, not gates.

## Sweep 4 — Honest MOSI

- Score **existing** MOSEI checkpoints on MOSI **test only**.
- Keep note 04 / the current CSVs as “merged corpus, historical.”
- Do not train on MOSI until the test-only table exists.

## Housekeeping I should do in the same sitting

Not model work, but it will save a weekend:

1. `state_dict` + json sidecar instead of `torch.save(model)`.
2. `MPLBACKEND=Agg` or delete `plt.show()` in `single_test`.
3. Fix `train_mosi_glove.py` import.
4. Set a seed at the top of every `if __name__` script.
5. Restore a real `total_epochs` in `train_main_bert.py`.
6. Rounded Acc-7 column next to uniform Acc-7.

I am listing these here so I do not “just add another fusion method”
when I sit down again.

## What I will not include in any of these sweeps

- Company data or internal checkpoints.
- A 12-layer text-only BERT fine-tune “to beat SOTA.”
- New datasets (IEMOCAP, podcasts) before sweep 1 is done.
