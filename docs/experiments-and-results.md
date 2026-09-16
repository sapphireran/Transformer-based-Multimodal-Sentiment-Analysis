# Experiments and results

All numbers below are copied from the CSVs already in this repository.
They are **personal run logs**, not a re-evaluated leaderboard. Scripts
under `model/` write the same column order:

```
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Acc-7 / Acc-5 use the **uniform** [-3, 3] bins described in
[training-and-evaluation.md](training-and-evaluation.md).

## File map

| CSV | Dataset | Text | What was swept |
| --- | --- | --- | --- |
| `model/results/main_results.csv` | MOSEI | BERT | six fusion baselines |
| `model/results/glove_results.csv` | MOSEI | GloVe | six fusion baselines |
| `model/results/ablation_results.csv` | MOSEI | BERT | GMTM modality subsets |
| `model/results/ablation_glove_results.csv` | MOSEI | GloVe | GMTM modality subsets |
| `model/mosi_test/mosi_bert_results.csv` | MOSI (merged) | BERT | MOSEI checkpoints + GMTM |
| `model/mosi_test/mosi_glove_results.csv` | MOSI (merged) | GloVe | same |
| `model/mosi_test/ablation_mosi_results.csv` | MOSI (merged) | BERT | GMTM subsets |
| `model/mosi_test/ablation_mosi_glove_results.csv` | MOSI (merged) | GloVe | GMTM subsets |

Duplicates at `model/main_results.csv`, `model/glove_results.csv`, and
`model/ablation_*.csv` match the `results/` copies.

`model/results/plot.ipynb` is a personal plotting notebook over these
tables. It is not required to read the numbers.

## MOSEI + BERT (main fusion sweep)

From `model/results/main_results.csv`:

| Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

Late transformer fusion is the best **baseline**. Low-rank tensor fusion
is the best classical (non-transformer) row and is cheaper than full
TensorFusion at almost the same quality.

Early vs. late concat are nearly tied. On this split, “when you fuse”
matters less than “whether you use a transformer or a low-rank product”.

## MOSEI + BERT (GMTM ablation)

From `model/results/ablation_results.csv`:

| Modalities | MAE | Acc-7 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.6293 | 0.2061 | 0.7693 |
| text + audio | 0.5659 | 0.4806 | 0.8409 | 0.7227 | 0.8731 |
| text + visual | 0.5667 | 0.4838 | 0.8324 | 0.7143 | 0.8697 |
| audio + visual | 0.8220 | 0.3951 | 0.6274 | 0.2231 | 0.7670 |
| text + audio + visual | **0.5640** | 0.4827 | **0.8429** | **0.7255** | **0.8777** |

Reading this as a personal note:

1. **Text is necessary.** Dropping it jumps MAE from ~0.56 to ~0.82 and
   kills correlation.
2. **Audio and visual are small additives** on top of BERT, not
   independent sentiment channels, at least with Facet / COVAREP.
3. GMTM all-three still beats TransformerLate (0.564 vs 0.585 MAE) and
   every concat / tensor row.
4. Audio+visual ≈ visual-only ≈ audio-only. The non-text streams do not
   rescue each other.

Because unused modalities are **zero-filled** rather than removed, a
text-only GMTM still spends parameters on two silent cross-attention
grids. The text-only number is therefore a lower bound on what a native
1-modality transformer would do.

## MOSEI + GloVe

Fusion sweep (`model/results/glove_results.csv`):

| Method | MAE | Acc-7 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.7975 | 0.6397 | 0.8425 |

Without contextual text, **LowRankTensorFusion** wins the baseline table.
TransformerLate no longer dominates. That matches the idea that the late
transformer was partly riding BERT’s already-contextual 768-d stream.

GMTM GloVe ablation (`model/results/ablation_glove_results.csv`):

| Modalities | MAE | Acc-2 | Corr |
| --- | ---: | ---: | ---: |
| text | 0.6616 | 0.8118 | 0.6557 |
| audio | 0.8232 | 0.6362 | 0.2067 |
| visual | 0.8139 | 0.6480 | 0.2256 |
| text + audio | 0.7200 | 0.7760 | 0.5503 |
| text + visual | 0.6612 | 0.7980 | 0.6459 |
| audio + visual | 0.8141 | 0.6489 | 0.2345 |
| text + audio + visual | **0.6251** | 0.8079 | **0.6714** |

Two GloVe-specific quirks worth remembering:

- `text+audio` is **worse** than text-only (MAE 0.720 vs 0.662). The extra
  zero-padded / noisy COVAREP stream can hurt when the language encoder is
  weaker.
- All-three GMTM (0.625) is close to LowRankTensorFusion (0.617) and no
  longer a clear winner.

## MOSI transfer (merged pool)

`get_mosi_dataloader` concatenates MOSI train+valid+test. These rows are
“MOSEI-trained checkpoint scored on all of MOSI”, not the official test
split.

BERT (`model/mosi_test/mosi_bert_results.csv`):

| Method | MAE | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.7604 | 0.6284 | 0.7704 |
| Concat (late) | 1.0056 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.7959** | **0.6748** | **0.7855** |
| GMTM | 0.9493 | 0.7638 | 0.6403 | 0.7626 |

GloVe (`model/mosi_test/mosi_glove_results.csv`):

| Method | MAE | Acc-2 | Corr |
| --- | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.6869 | 0.4774 |
| Concat (late) | 1.0797 | 0.7064 | 0.5520 |
| LowRankTensorFusion | 1.0825 | 0.7336 | 0.5559 |
| TensorFusion | 1.1693 | 0.6737 | 0.4958 |
| TransformerEarly | 1.1941 | 0.6938 | 0.4822 |
| TransformerLate | 1.0855 | 0.7245 | 0.5303 |
| GMTM | **0.9748** | **0.7355** | **0.6177** |

MOSI is a harsher, smaller domain. Absolute MAE is much higher than MOSEI.
On BERT, TransformerLate transfers best; GMTM does not. On GloVe, GMTM
does transfer best. Do not average those two sentences into “GMTM always
wins”.

MOSI GMTM ablations repeat the MOSEI story: text dominates; audio / visual
alone are near chance correlation (BERT audio Corr **−0.13**).

## Practical takeaways for later personal runs

1. Report MOSEI BERT all-three GMTM as the primary number (0.564 MAE).
2. Always include a LowRankTensorFusion baseline — it is the one to beat
   when text is GloVe.
3. Never claim audio/visual “work” from Acc-2 alone; look at Corr.
4. If you publish MOSI numbers, re-evaluate on the **official test split**
   (`get_dataloader`, not `get_mosi_dataloader`).
5. Keep BERT and GloVe tables separate. Pooling them hides the fusion
   ranking reversal.
