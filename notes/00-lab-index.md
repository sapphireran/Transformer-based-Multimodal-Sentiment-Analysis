# Lab index

Personal notes for the Transformer / GMTM multimodal sentiment runs.
I am writing these after the fact, from the CSVs in `model/` plus a re-read of
the scripts. Dates on the notebooks cluster around late 2024.

Nothing here is a paper claim. Several evaluation choices (uniform Acc-7, MOSI
split merge, GMTM zero-fill ablations) make these numbers **incomparable** to
the usual leaderboard without a re-score.

## Experiment log

| Note | Dataset | Text | Model | Artifact |
| ---- | ------- | ---- | ----- | -------- |
| [01](01-mosei-bert-fusion.md) | MOSEI in-domain | BERT 768 | 6 fusion methods | `model/main_results.csv` |
| [02](02-mosei-glove-fusion.md) | MOSEI in-domain | GloVe 300 | 6 fusion methods | `model/glove_results.csv` |
| [03](03-gmtm-ablation-mosei.md) | MOSEI in-domain | BERT + GloVe | GMTM, 7 modality subsets | `model/ablation_*.csv` |
| [04](04-mosi-transfer.md) | MOSI, **merged splits** | BERT + GloVe | MOSEI ckpts, fusion + GMTM | `model/mosi_test/*.csv` |
| [05](05-gmtm-design-notes.md) | — | — | Why GMTM looks the way it does | `model/models.py` |
| [06](06-hyperparams-and-training-protocol.md) | both | both | Optimizer, epochs, packing | train_*.py |
| [07](07-failure-modes-and-open-questions.md) | — | — | Bugs, dead knobs, next runs | — |
| [08](08-reading-list-and-personal-context.md) | — | — | What I was reading, what I am not doing | — |
| [09](09-numeric-appendix.md) | both | both | All CSV floats in one grep-able page | every `*.csv` |

## Working conclusions I still believe

1. **Text is the task.** On MOSEI BERT, GMTM text-only is already `MAE 0.5687`,
   Acc-2 `0.8404`. Full trimodal is `0.5640` / `0.8429`. The extra modalities
   are a ~0.5% MAE relative move.
2. **Among fusion methods with BERT, late transformer won.** `TransformerLate`
   `MAE 0.5846` vs LMF `0.5970` vs early concat `0.6178`.
3. **GloVe changes the ranking.** LMF wins the GloVe bake-off. Late transformer
   is only mid-pack. Weaker text makes low-rank multiplicative fusion look better.
4. **Audio and vision alone do not have the label.** MOSEI BERT audio-only Corr
   `0.1124`. Vision-only `0.2061`. A+V without text stays in that band.
5. **GloVe + audio can *hurt*.** MOSEI GMTM `text+audio` is worse than text-only
   (`MAE 0.7200` vs `0.6616`). Same pattern on MOSI GloVe transfer.
6. **MOSI transfer is a vibe check, not a score.** The loader concatenates all
   MOSI splits. BERT late transformer still looks best among the six; GMTM only
   wins the GloVe transfer table.

## How I want to use this folder

When I rerun something, append a short dated section to the relevant note rather
than creating a parallel file. If a CSV changes, say so in the same note and
leave the old table in place with a strike or a “superseded” line.

Do not put company work in this repo. MOSI/MOSEI + this stack only.
