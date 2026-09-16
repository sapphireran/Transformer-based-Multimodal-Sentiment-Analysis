# How to read the committed CSVs

The tables in [results.md](results.md) are **single-run** snapshots, not
seed averages. Use them to reason about this checkout, not as a leaderboard
submission.

## What “best” means

Lower MAE is the training objective. Acc-7 / Acc-5 / Acc-2 / Corr / F1 are
derived from the same scalar predictions (see [metrics.md](metrics.md)).
A method can win Acc-2 and lose MAE; that happened on MOSEI GloVe (LMF vs
GMTM). Starred cells from `examples/inspect_results.py` pick the winner
**per column**, independently.

## Text does the work

On MOSEI BERT GMTM ablation:

| Slice | MAE | Corr |
| --- | --- | --- |
| text | 0.5687 | 0.7202 |
| audio | 0.8306 | 0.1124 |
| visual | 0.8217 | 0.2061 |
| text+audio+visual | 0.5640 | 0.7255 |

Audio or visual alone is close to uninformative on correlation. Adding
them to text still helps a little (MAE 0.5687 → 0.5640). Any story that
starts with “the gated transformer discovered rich non-verbal affect”
has to survive this table first.

GloVe repeats the pattern, with a warning: **text+audio is worse than
text-only** (MAE 0.7200 vs 0.6616). Zeroing a stream is not the same as
retraining a two-input graph; noise in a live encoder can hurt.

## BERT vs GloVe

Same fusion zoo, weaker text. Every GloVe row is worse MAE than its BERT
twin. GMTM’s GloVe MAE (0.6251) is in the same band as BERT ConcatEarly
(0.6178). If you only have GloVe pickles, LMF is a strong cheap baseline.

## MOSEI vs MOSI transfer

`get_mosi_dataloader` **concatenates MOSI train+valid+test** and scores
that union with a MOSEI-trained checkpoint. That is a transfer / domain
shift probe, not the MOSI test protocol from the original papers.

On that probe, **Transformer-late wins MOSI BERT** (MAE 0.8986 vs GMTM
0.9493). GMTM **does** win MOSI GloVe. Do not quote MOSI numbers from this
repo as “MOSI test Acc-7”.

## Uniform Acc-7 is not integer Acc-7

Bins are equal-width slices of `[-3, 3]` (`6/7 ≈ 0.857` wide). Older MOSI
papers often treat the annotator’s `{−3,…,3}` as seven classes directly.
A 0.48 Acc-7 here is not interchangeable with a 0.48 from those tables.
`examples/bin_edges.py` prints the edges; `examples/metric_sensitivity.py`
shows how a 0.1 shift can drop Acc-7 while MAE barely moves.

## Plotting

`model/results/plot.ipynb` concatenates the last GMTM ablation row onto
the six-fusion tables and draws BERT vs GloVe lines. The headless
stand-in is `examples/plot_results.py`, which writes PNGs under
`examples/output/` (gitignored).
