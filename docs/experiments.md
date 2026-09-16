# Experiment catalog

Tables below are copied from the CSVs already in the repo. I did not
re-train these runs in this docs pass. Use them as a map of what each
script was meant to produce.

## Script → artifact

| Script | Data | Text | Writes |
| --- | --- | --- | --- |
| `model/train_main_bert.py` | MOSEI | BERT | `model/main_results.csv`, `checkpoints/{Fusion}.pt` |
| `model/train_main_glove.py` | MOSEI | GloVe | `model/glove_results.csv`, `checkpoints/glove_{Fusion}.pt` |
| `model/train_GMTM_bert.py` | MOSEI | BERT | `model/ablation_results.csv`, `checkpoints/ablation/model_{mods}.pt` |
| `model/train_GMTM_glove.py` | MOSEI | GloVe | `model/ablation_glove_results.csv`, `checkpoints/ablation/model_glove_{mods}.pt` |
| `model/mosi_test/train_mosi_bert.py` | MOSI (eval) | BERT | `mosi_bert_results.csv` |
| `model/mosi_test/train_mosi_glove.py` | MOSI (eval) | GloVe | `mosi_glove_results.csv` |
| `model/mosi_test/mult_bert_mosi.py` | MOSI (eval) | BERT | GMTM row / `ablation_mosi_results.csv` |
| `model/mosi_test/mult_glove_mosi.py` | MOSI (eval) | GloVe | GMTM glove MOSI ablation |

Duplicates under `model/results/` are the same tables kept next to the
plot notebook.

## MOSEI · BERT · fusion sweep

From [`model/results/main_results.csv`](../model/results/main_results.csv).

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |

TransformerLate is the best row on every column.

## MOSEI · GloVe · fusion sweep

From [`model/results/glove_results.csv`](../model/results/glove_results.csv).

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | 0.6174 | 0.4458 | 0.5287 | 0.8107 | 0.6704 | 0.8523 |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

GloVe is uniformly worse than BERT. Low-rank tensor fusion wins this
column, not TransformerLate.

## MOSEI · BERT · GMTM modality ablation

From [`model/results/ablation_results.csv`](../model/results/ablation_results.csv).
Dropped modalities are zeros, not removed encoders.

| Modalities | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | 0.5640 | 0.4827 | 0.5561 | 0.8429 | 0.7255 | 0.8777 |

Reading this as a personal note: text already beats every fusion
baseline's MAE (0.5687 vs TransformerLate 0.5846). Adding audio and
visual on top of BERT is a small but consistent lift. Non-text pairs
do not.

## MOSEI · GloVe · GMTM modality ablation

From [`model/results/ablation_glove_results.csv`](../model/results/ablation_glove_results.csv).

| Modalities | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | 0.6251 | 0.4467 | 0.5317 | 0.8079 | 0.6714 | 0.8474 |

GloVe GMTM *does* benefit from all three streams (0.625 vs 0.662 text
only). The `text+audio` row regresses — worth a rerun before treating
it as a real interaction effect.

## MOSI transfer · BERT

From [`model/mosi_test/mosi_bert_results.csv`](../model/mosi_test/mosi_bert_results.csv).
MOSEI checkpoints, MOSI loader (merged splits).

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | 0.8986 | 0.3230 | 0.3894 | 0.7959 | 0.6748 | 0.7855 |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

## MOSI transfer · GloVe

From [`model/mosi_test/mosi_glove_results.csv`](../model/mosi_test/mosi_glove_results.csv).

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GatedMultiTransfomer | 0.9748 | 0.3152 | 0.3995 | 0.7355 | 0.6177 | 0.7381 |

On GloVe→MOSI, GMTM is the only row that stays near 0.97 MAE / 0.32 Acc7.

## MOSI · GMTM ablations

BERT: [`ablation_mosi_results.csv`](../model/mosi_test/ablation_mosi_results.csv).
GloVe: [`ablation_mosi_glove_results.csv`](../model/mosi_test/ablation_mosi_glove_results.csv).

Same story as MOSEI: text dominates; audio/visual-only Corr is near 0
or negative. The full triple is not always the best MOSI BERT row
(text+visual MAE 0.9044 vs triple 0.9493).

## How I read the whole grid

1. **Text embedding quality dwarfs fusion choice.** BERT vs GloVe is a
   larger gap than Concat vs Transformer.
2. **On BERT-MOSEI, late transformer fusion is the strongest baseline.**
   GMTM's value shows up more clearly in the GloVe and MOSI-transfer
   settings than on the BERT-MOSEI leaderboard.
3. **Unimodal non-text is weak.** Any claim that "video helps" has to
   be a *delta on top of text*, not a unimodal number.
4. **Do not mix loader conventions.** MOSEI test is the official test
   split; MOSI scripts merge splits. Quote them separately.

## Plot notebook

`model/results/plot.ipynb` is the place that turns these CSVs into
figures. It is not executed in the synthetic example suite.
