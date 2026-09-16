# Results

Numbers below are copied from the CSVs already in the tree. They are **not**
re-fit by the examples. Use them as a reading of the personal experiment log.

Sources:

- [`model/results/main_results.csv`](../model/results/main_results.csv)
- [`model/results/glove_results.csv`](../model/results/glove_results.csv)
- [`model/results/ablation_results.csv`](../model/results/ablation_results.csv)
- [`model/results/ablation_glove_results.csv`](../model/results/ablation_glove_results.csv)
- [`model/mosi_test/mosi_bert_results.csv`](../model/mosi_test/mosi_bert_results.csv)
- [`model/mosi_test/mosi_glove_results.csv`](../model/mosi_test/mosi_glove_results.csv)
- [`model/mosi_test/ablation_mosi_results.csv`](../model/mosi_test/ablation_mosi_results.csv)
- [`model/mosi_test/ablation_mosi_glove_results.csv`](../model/mosi_test/ablation_mosi_glove_results.csv)

Metric definitions: [`metrics.md`](metrics.md). MOSI protocol warning:
[`data-pipeline.md`](data-pipeline.md) (merged splits).

`examples/read_result_tables.py` pretty-prints the same files.

## MOSEI · BERT text

Baseline fusion:

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |

GMTM modality ablation:

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

Reading:

- **TransformerLate** is the strongest *baseline* (best MAE / Acc-7 / Corr / F1
  in the six-way sweep).
- **GMTM full** beats that baseline on every recorded column.
- **Text dominates.** Text-only GMTM is within `0.005` MAE of the full model.
  Audio-only and visual-only sit near the majority-ish error (`MAE ≈ 0.82`)
  with near-zero correlation.
- Adding audio or visual to text is a small, noisy gain. Acc-7 is actually
  highest for text+visual (`0.4838`), while MAE / Acc-2 / Corr / F1 prefer the
  full trio. Do not over-claim complementarity from a single seed.

## MOSEI · GloVe text

Baseline fusion:

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | 0.6174 | 0.4458 | 0.5287 | 0.8107 | 0.6704 | 0.8523 |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

GMTM modality ablation:

| Modalities | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

Reading:

- GloVe is strictly weaker than BERT for every comparable full-model row
  (GMTM MAE `0.6251` vs `0.5640`).
- With GloVe, **LMF** is the strongest baseline, not TransformerLate.
- text+audio **hurts** relative to text-only (MAE `0.7200`, Acc-7 `0.3503`).
  That is a useful negative result: naively adding a weak stream can inject
  noise, especially when the unused-modality path is zeros rather than an
  absent module.
- The full trio recovers and is the best GloVe GMTM row on MAE / Acc-7 /
  Acc-5 / Corr.

## MOSI transfer (merged splits) · BERT

Checkpoints trained on MOSEI, scored on **all** MOSI clips.

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

GMTM BERT ablation on the same merged MOSI data:

| Modalities | MAE ↓ | Acc-7 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| text | 0.9363 | 0.3170 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.1681 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.1663 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.3055 | 0.6494 | 0.7715 |
| text+visual | **0.9044** | **0.3257** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.1654 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.3078 | 0.6403 | 0.7626 |

Transfer is a different story than in-domain MOSEI: **TransformerLate**
generalizes better than full GMTM, and **text+visual** GMTM beats the full
trio. Audio without text is anti-correlated. Treat this as “MOSEI-trained
models on a smaller, merged MOSI pool”, not as a MOSI leaderboard.

## MOSI transfer (merged splits) · GloVe

| Fusion | MAE ↓ | Acc-7 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.6869 | 0.4774 | 0.6840 |
| Concat | 1.0797 | 0.2267 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.7245 | 0.5303 | 0.7271 |
| GatedMultiTransfomer | **0.9748** | **0.3152** | **0.7355** | **0.6177** | **0.7381** |

Here GMTM **does** win the GloVe transfer sweep, and the matching ablation
row for the full trio is the same `0.9748` MAE. Text-only is close
(`1.0085`); text+audio again collapses (`1.2863`), same pathology as MOSEI
GloVe.

## Practical takeaways

1. **BERT + GMTM + all three streams** is the best in-domain MOSEI setup in
   this log.
2. **Text is necessary.** Any row without text is not a usable sentiment
   model under these features.
3. **GloVe is a strictly weaker text tower** for this pipeline; LMF is its
   best simple baseline, GMTM its best full model.
4. **Transfer to MOSI is fragile** and uses a non-standard split. Quote those
   numbers only with the protocol attached.
5. Single-seed CSVs, mixed commented-out training calls, and at least one
   signature mismatch (`train_GMTM_glove.py` vs `test()`) mean this is a
   research notebook in script form — not a locked training recipe. The
   examples exist so the *modules and metrics* stay inspectable even when the
   GPU jobs do not.
