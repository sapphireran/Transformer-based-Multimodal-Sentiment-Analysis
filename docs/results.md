# Results

Numbers below are copied from the CSVs already in the repo. They are **not** re-computed in this documentation pass. Use `python examples/summarize_results.py` to reprint them from the files.

## MOSEI, BERT text (main fusion sweep)

Source: `model/results/main_results.csv`

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |

Late transformer fusion is the best of the six baselines on every column except Acc-5, where it is also best. Low-rank tensor fusion is the best non-transformer baseline. Early and late concat are nearly tied and clearly worse.

## MOSEI, BERT GMTM modality ablation

Source: `model/results/ablation_results.csv`

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

Reading this table as a personal study:

- **Text is almost the entire task.** Unimodal text already beats every fusion baseline in the main sweep (MAE 0.5687 vs. late transformer 0.5846).
- **Audio-only and visual-only are weak** (Corr 0.11 / 0.21). They barely beat a constant predictor on ranking.
- **Audio+visual without text** does not recover (MAE 0.8220) — the two non-text streams do not substitute for words.
- Adding audio and/or visual on top of text helps a little (0.5687 → 0.5640 MAE, +0.005 Corr). The gain is real but small.
- Acc-7 peaks on **text+visual** (0.4838), not on the full trio. MAE / Corr / F1 still prefer the full trio.

GMTM with all three modalities is the best BERT / MOSEI result in the repo.

## MOSEI, GloVe text (main fusion sweep)

Source: `model/results/glove_results.csv`

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | 0.5287 | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

Switching BERT → GloVe **changes the winner**. Low-rank tensor fusion is best; late transformer is mid-pack. Early concat / early transformer suffer the most from the weaker text encoder.

## MOSEI, GloVe GMTM modality ablation

Source: `model/results/ablation_glove_results.csv`

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

Unlike BERT, GloVe GMTM **needs all three modalities** to beat unimodal text on MAE (0.6616 → 0.6251). The `text+audio` row is a regression (MAE 0.7200, Acc-7 0.3503) — worth treating as an unstable ablation rather than a real “audio hurts” finding until it is rerun.

Full GMTM GloVe (0.6251 MAE) is close to LRTF GloVe (0.6174). The custom transformer does **not** dominate the GloVe sweep the way it dominates BERT.

## MOSI transfer, BERT (MOSEI checkpoints)

Source: `model/mosi_test/mosi_bert_results.csv`. Evaluation pool is **merged MOSI**, not the official test split ([evaluation](evaluation.md)).

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat (late) | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GMTM | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

Transfer MAE is ~0.3 worse than in-domain MOSEI, as expected (MOSI is smaller / more speaker-idiosyncratic). Late transformer transfers best. GMTM is mid-pack on this particular transfer CSV — it does **not** keep the in-domain crown.

## MOSI transfer, BERT GMTM ablation

Source: `model/mosi_test/ablation_mosi_results.csv`

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.9363 | 0.3170 | 0.4127 | 0.7667 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.1681 | 0.1842 | 0.4509 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.1663 | 0.1837 | 0.5165 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.3055 | 0.3995 | 0.7786 | 0.6494 | 0.7715 |
| text+visual | **0.9044** | **0.3257** | 0.4104 | **0.7796** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.1654 | 0.1846 | 0.5208 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

On MOSI transfer, **text+visual beats the full trio**. Audio-only / visual-only correlations go negative or near zero — those MOSEI-trained non-text towers do not travel.

## MOSI transfer, GloVe

Source: `model/mosi_test/mosi_glove_results.csv`

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat (late) | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GMTM | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

Here GMTM **does** win the GloVe MOSI transfer table, and by a wide MAE margin (0.97 vs ~1.08). That is the opposite of the BERT MOSI transfer picture.

## MOSI transfer, GloVe GMTM ablation

Source: `model/mosi_test/ablation_mosi_glove_results.csv`

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 1.0085 | 0.3014 | 0.3903 | 0.7317 | 0.5959 | 0.7343 |
| audio | 1.3799 | 0.1681 | 0.1842 | 0.4696 | −0.0720 | 0.5513 |
| visual | 1.3692 | 0.1667 | 0.1832 | 0.5036 | 0.0409 | 0.6177 |
| text+audio | 1.2863 | 0.1745 | 0.2400 | 0.6033 | 0.3524 | 0.5141 |
| text+visual | 1.0057 | 0.2739 | 0.3729 | 0.7235 | 0.5967 | 0.7446 |
| audio+visual | 1.3734 | 0.1672 | 0.1837 | 0.4998 | −0.0042 | 0.6024 |
| text+audio+visual | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | 0.7381 |

Full trio is best on MAE / Acc-7 / Corr. `text+audio` is again a bad row (same pattern as in-domain GloVe).

## Personal takeaways

1. **BERT + GMTM + all modalities** is the best in-domain MOSEI result (MAE 0.5640, Corr 0.7255).
2. **Text is necessary.** Every audio/visual-only row is in a different quality band.
3. **Which fusion wins depends on the text encoder.** BERT favors late transformers / GMTM; GloVe favors low-rank tensor fusion in-domain and GMTM on MOSI transfer.
4. **Do not quote MOSI CSVs as standard MOSI test.** They are transfer-on-merged-splits.
5. A few ablation rows (`text+audio` GloVe) look unstable. Rerun before building a paper plot on them.

`model/results/plot.ipynb` concatenates the GMTM full-trio row onto the main BERT/GloVe tables under the name `GatedMultiTransformer` and is the personal plotting entry point.
