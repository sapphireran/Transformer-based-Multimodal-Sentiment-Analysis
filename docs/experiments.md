# Experiment log

All tables below are copied from the CSV files already in this repo.
They are **single-run** numbers from the original machine, not a mean ±
std over seeds. Read them as a personal ranking, not as a leaderboard
submission.

Scripts that produced them:

| Table | Script | Data |
| --- | --- | --- |
| MOSEI BERT fusion | `model/train_main_bert.py` | `mosei_raw_bert.pkl` |
| MOSEI GloVe fusion | `model/train_main_glove.py` | `mosei_raw_glove.pkl` |
| MOSEI BERT GMTM ablation | `model/train_GMTM_bert.py` | same BERT pickle, zero-masked mods |
| MOSEI GloVe GMTM ablation | `model/train_GMTM_glove.py` | same GloVe pickle |
| MOSI transfer, BERT | `model/mosi_test/train_mosi_bert.py` | MOSEI ckpt → merged MOSI |
| MOSI transfer, GloVe | `model/mosi_test/train_mosi_glove.py` | same idea |
| MOSI GMTM ablation | `model/mosi_test/mult_*_mosi.py` | MOSEI GMTM ckpt → MOSI |

Duplicates under `model/*.csv` and `model/results/*.csv` match for the
main MOSEI tables. Prefer `model/results/` as the tidy copy.

## 1. MOSEI, BERT text — fusion sweep

Source: [`model/results/main_results.csv`](../model/results/main_results.csv)

| Fusion | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

Late transformer fusion wins every column in this sweep. LMF is the best
*non-transformer* tensor method and beats full TFN — expected, because TFN
pays for a 128k-wide outer product that the head then has to compress.

Early concat ≈ late concat. Once BERT is present, “where you concat” is
less important than “whether you let modalities attend.”

## 2. MOSEI, GloVe text — fusion sweep

Source: [`model/results/glove_results.csv`](../model/results/glove_results.csv)

| Fusion | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | **0.5287** | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

With 300-d GloVe, **LMF wins**. TransformerLate no longer leads. Two
plausible reasons, both consistent with the code:

- Late GloVe `in_dim=1792` does not match `64+128+512=704` (see quirks).
- Static embeddings give the transformer less language signal, so a
  multiplicative fusion of GRU summaries can be the better inductive bias.

Every GloVe row is worse than the matching BERT row. The text encoder
matters more than the fusion logo on the slide.

## 3. MOSEI, BERT — GMTM modality ablation

Source: [`model/results/ablation_results.csv`](../model/results/ablation_results.csv)

| Modalities | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | **0.5596** | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | **0.4838** | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

What this ablation is allowed to say:

- Text is necessary. Non-text pairs stay near Acc2 ≈ 0.63 and Corr ≤ 0.23.
- Trimodal GMTM is the best MAE / Acc2 / Corr / F1 in the whole BERT stack,
  including TransformerLate (0.5846 MAE).
- The trimodal vs. text-only MAE gap is **0.0047**. That is a real
  direction but a small one. Acc2 moves `0.8404 → 0.8429`.
- text+visual *loses* Acc2 and F1 vs. text-only, while winning Acc7. The
  extra visual stream is not a free lunch on every metric.

## 4. MOSEI, GloVe — GMTM modality ablation

Source: [`model/results/ablation_glove_results.csv`](../model/results/ablation_glove_results.csv)

| Modalities | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | **0.8118** | 0.6557 | **0.8509** |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

GloVe trimodal GMTM **does** beat GloVe text-only on MAE and Corr, but
**loses** Acc2 and F1. text+audio is actively harmful (MAE 0.7200). This
is the table that most clearly says “more modalities ≠ automatically
better,” especially with a weaker language feature.

Trimodal GloVe GMTM (0.6251 MAE) is in the same band as GloVe LMF (0.6174)
from the main sweep — fusion choice and text features interact.

## 5. MOSI transfer, BERT (MOSEI checkpoints)

Source: [`model/mosi_test/mosi_bert_results.csv`](../model/mosi_test/mosi_bert_results.csv)

Evaluation pool = MOSI train+valid+test concatenated. Not the official
MOSI test set.

| Fusion | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

Domain shift is large (MAE jumps from ~0.58 on MOSEI to ~0.90 on MOSI).
TransformerLate still leads the fusion family. GMTM is mid-pack here —
the MOSEI-best model is not automatically the MOSI-best transfer model.

## 6. MOSI transfer, GloVe

Source: [`model/mosi_test/mosi_glove_results.csv`](../model/mosi_test/mosi_glove_results.csv)

| Fusion | MAE ↓ | Acc7 ↑ | Acc5 ↑ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GatedMultiTransfomer | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

On GloVe MOSI transfer, GMTM is the clear winner. Combined with table 4,
GMTM looks more helpful when the language stream is weaker.

## 7. MOSI transfer — GMTM ablation (BERT)

Source: [`model/mosi_test/ablation_mosi_results.csv`](../model/mosi_test/ablation_mosi_results.csv)

| Modalities | MAE ↓ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| text | 0.9363 | 0.7667 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.4509 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.5165 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.7786 | 0.6494 | 0.7715 |
| text+visual | **0.9044** | **0.7796** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.5208 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.7638 | 0.6403 | 0.7626 |

Audio-only Corr is **negative**. Trimodal is *worse* than text+visual.
For MOSI transfer with BERT, dropping audio is the better personal default.

## 8. MOSI transfer — GMTM ablation (GloVe)

Source: [`model/mosi_test/ablation_mosi_glove_results.csv`](../model/mosi_test/ablation_mosi_glove_results.csv)

| Modalities | MAE ↓ | Acc2 ↑ | Corr ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| text | 1.0085 | 0.7317 | 0.5959 | 0.7343 |
| audio | 1.3799 | 0.4696 | −0.0720 | 0.5513 |
| visual | 1.3692 | 0.5036 | 0.0409 | 0.6177 |
| text+audio | 1.2863 | 0.6033 | 0.3524 | 0.5141 |
| text+visual | 1.0057 | 0.7235 | 0.5967 | 0.7446 |
| audio+visual | 1.3734 | 0.4998 | −0.0042 | 0.6024 |
| text+audio+visual | **0.9748** | **0.7355** | **0.6177** | 0.7381 |

Here trimodal GloVe GMTM *does* win MAE / Acc2 / Corr. Same model family,
same MOSI pool, opposite “should I keep audio?” answer once the text
feature changes. That interaction is the most useful sentence in this log.

## Working conclusions (personal)

1. **Use BERT if you can.** It beats GloVe in every matched cell on MOSEI.
2. **Fusion ranking depends on the text feature.** TransformerLate for BERT;
   LMF (main sweep) or GMTM (ablation / MOSI transfer) for GloVe.
3. **Text is the load-bearing modality.** Publish a text-only number next
   to every multimodal claim.
4. **Do not average MOSI-transfer CSVs with official MOSI test papers.**
   The loader merges splits.
5. **Single seed.** A 0.005 MAE gap is a hint, not a theorem.
