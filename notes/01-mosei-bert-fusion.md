# 01 — MOSEI BERT fusion bake-off

**Script:** `model/train_main_bert.py`
**Data:** `data/MOSEI/mosei_raw_bert.pkl`
**CSV:** `model/main_results.csv` (duplicate under `model/results/`)
**Checkpoints:** `model/checkpoints/{FusionMethod}.pt` (local only)
**Text:** BERT 768-d, visual 35, audio 74, batch 32

## Why I ran this

I wanted a single table of “obvious” fusion choices on the same loader, loss, and
optimizer before I spent more time on GMTM. The six methods are the ones I already
had implementations for from the MultiBench-style modules in `models.py`.

Question going in: does a transformer fusion beat TFN / LMF on MOSEI when the
text front is already BERT?

## Table (copied from the CSV)

| Fusion Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| ------------- | ---: | ----: | ----: | ----: | ---: | --: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

Acc-7 / Acc-5 are **uniform bins**, not rounded integers. See `docs/metrics.md`.

## What I think the ranking is saying

**Late transformer is the only method that is best on every column.** That is
cleaner than I expected. The MAE gap vs LMF is 0.0124, vs early concat 0.0332.
On Acc-2 the spread from worst to best is about 3.1 points (0.8084 → 0.8393).

**Early vs late concat are a tie for last.** 0.6178 vs 0.6148 MAE, identical
Acc-2. Giving each modality its own LSTM did not help polarity, and barely
helped MAE. With BERT, a joint LSTM on 877-d is not obviously worse than three
separate LSTMs. I read that as “the recurrent encoder is not the bottleneck;
fusion of summaries vs joint time is also not the bottleneck — both concat
stories are just weak.”

**LMF > TFN.** Same GRU+linear encoder family, much smaller head. Full tensor
fusion’s 128000-d MLP is a regularization nightmare at this batch size. LMF’s
rank-32 factorization is the better inductive bias here, and it shows up on
Corr (0.6925 vs 0.6776) as well as MAE.

**Early transformer is mid-pack, closer to TFN than to late transformer.**
I concat raw 877-d frames, project to 32-d, and take the last step. Two
problems I already know about: `embed_dim=32` is small for BERT-scale text,
and the `batch_first=True` / permute mismatch in `EarlyFusionTransformer`.
I do **not** know how much of the 0.021 MAE gap vs late transformer is the
idea vs the implementation.

**Late transformer has the right shape for this data.** Per-modality
transformers (64 / 128 / 1024) let text keep a fat residual stream. Fusion
then happens in a 1216-d sequence at 32-d after the second conv. Text is
allowed to be “the wide one” until the last projection. That matches the
ablation result that text is doing almost all the work.

## Protocol caveats for this table

- `train_main_bert.py` as committed trains **1 epoch**. I do not believe the
  CSV is a 1-epoch run. Either I changed the epoch count after logging, or I
  ran a different copy. Treat the script as the architecture wiring, not the
  recipe that produced these exact floats.
- TransformerEarly uses the max-pad loader; everyone else uses packed
  sequences. So early transformer is also a **batching** change.
- No seed is set in the script. This is a single run per method.
- Checkpoints are full-module pickles.

## Compared to GMTM on the same front

GMTM full trimodal BERT (`ablation_results.csv`) is `MAE 0.5640`, Acc-2
`0.8429`, Corr `0.7255`. That is better MAE / Corr than TransformerLate
(`0.5846` / `0.7041`) and only slightly better Acc-2 (`0.8429` vs `0.8393`).

I am not calling that a fair architecture contest. GMTM uses max-pad +
Identity encoders + a 3×3 crossmodal grid; TransformerLate uses packed
per-modality transformers + a second transformer. Different capacity,
different padding. The useful sentence is: **both transformer-ish fusions
beat concat and TFN on this BERT table, and GMTM’s extra crossmodal grid
shows up more in MAE/Corr than in Acc-2.**

## What I would do if I reran this bake-off

1. Set `total_epochs` to 20–50, early stop 7, three seeds.
2. Fix early transformer `batch_first`.
3. Log valid MAE each epoch so I can see whether late transformer wins
   because it trains, or because it starts better.
4. Add a text-only LSTM baseline in the same script. Without it, I cannot
   tell how much of TransformerLate’s 0.5846 is fusion vs “a wide text
   transformer with some extra channels.”
5. Write `state_dict` + hparams, not `torch.save(model)`.

## Personal takeaway

On MOSEI BERT I should stop investing in concat and full TFN. LMF is the
non-transformer baseline to beat. Late transformer is the transformer
baseline to beat. GMTM has to clear **both**, on MAE *and* on a text-only
control, before I call the gates useful.
