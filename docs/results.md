# Results

Numbers below are copied from the CSVs already in this repository.
They come from the personal MOSI / MOSEI runs checked in under
`model/` and `model/results/`. `examples/inspect_results.py` reprints
the same tables from disk so a future overwrite is easy to spot.

## MOSEI · BERT features · fusion bake-off

Source: `model/results/main_results.csv` (also `model/main_results.csv`).

| Fusion Method | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

Among the six generic fusion graphs, late transformer fusion is the
best on every column. Low-rank tensor fusion is the strongest
non-transformer baseline.

## MOSEI · BERT features · GMTM modality ablation

Source: `model/ablation_results.csv`.

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | **0.5596** | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | **0.4838** | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

Read:

- Text alone already beats every generic fusion row in the bake-off.
- Audio or visual alone is weak (Corr 0.11–0.21). Together they still
  do not approach text.
- Adding audio and/or visual to text is a small but consistent MAE /
  Corr gain. The full trio is the best GMTM row.

## MOSEI · GloVe features · fusion bake-off

Source: `model/results/glove_results.csv`.

| Fusion Method | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | **0.5287** | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

With 300-D GloVe, LRTF overtakes TransformerLate. Every score is worse
than the matching BERT row — the text tower is the main lever.

## MOSEI · GloVe features · GMTM ablation

Source: `model/ablation_glove_results.csv`.

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

GloVe GMTM is more brittle: text+audio is *worse* than text alone.
The full trio recovers and posts the best MAE / Acc-7 / Corr, slightly
behind LRTF on Acc-2 / F1.

## MOSI transfer · BERT (MOSEI checkpoints)

Source: `model/mosi_test/mosi_bert_results.csv`. MOSI is smaller and
the loaders evaluate the union of its splits.

| Fusion Method | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

## MOSI transfer · BERT · GMTM ablation

Source: `model/mosi_test/ablation_mosi_results.csv`.

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.9363 | 0.3170 | **0.4127** | 0.7667 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.1681 | 0.1842 | 0.4509 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.1663 | 0.1837 | 0.5165 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.3055 | 0.3995 | 0.7786 | 0.6494 | 0.7715 |
| text+visual | **0.9044** | **0.3257** | 0.4104 | **0.7796** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.1654 | 0.1846 | 0.5208 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

On MOSI-BERT the text+visual pair beats the full trio. Audio is
actively harmful in this transfer setting (negative correlation when
used alone).

## MOSI transfer · GloVe

Source: `model/mosi_test/mosi_glove_results.csv` and
`ablation_mosi_glove_results.csv`.

Bake-off winner: **GatedMultiTransfomer** at MAE 0.9748 / Acc-7 0.3152 /
Corr 0.6177 — the only GloVe-MOSI row under 1.0 MAE. The next best
generic fusion is ConcatLate at 1.0797.

Ablation winner is the full trio (same 0.9748 row). Text-only is
1.0085; any audio-only or visual-only run sits around 1.37 MAE with
near-chance Acc-7 (~0.17).

## Short takeaways

1. **BERT >> GloVe** on both corpora for every comparable graph.
2. **Text is the backbone.** Non-text pairs never get close.
3. **On MOSEI-BERT**, GMTM (full) has the best MAE / Corr / Acc-2 / F1
   in the repo; TransformerLate wins the generic bake-off.
4. **On MOSEI-GloVe**, LRTF wins the bake-off; GMTM full is close on
   MAE and slightly better on Acc-7.
5. **MOSI transfer is much harder** (MAE jumps from ~0.56 to ~0.90).
   TransformerLate generalizes best among BERT fusion graphs; GMTM
   prefers dropping audio (text+visual).
6. Plots for some of these tables were drafted in
   `model/results/plot.ipynb` (large notebook, local figures).
