# Recorded results

Numbers below are copied from the CSVs under `model/results/` and
`model/mosi_test/`. They are **personal experimental runs**, not a claim
about a public leaderboard. Metrics are defined in
[evaluation.md](evaluation.md).

Best cell in each column is bolded when a single winner is obvious.

## MOSEI — BERT text (fusion sweep)

Source: `model/results/main_results.csv` (`train_main_bert.py`).

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | **0.5846** | **0.4675** | **0.5460** | **0.8393** | **0.7041** | **0.8699** |

Late transformer fusion wins every column. LMF is the best non-transformer
row and beats full TFN, which matches the usual “low-rank is enough”
story.

Early concat ≈ late concat. Sharing one LSTM over `877` features does not
beat three unimodal LSTMs here.

## MOSEI — GloVe text (fusion sweep)

Source: `model/results/glove_results.csv`.

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | **0.4458** | **0.5287** | **0.8107** | **0.6704** | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

With 300-D GloVe, **LMF wins the sweep**. TransformerLate no longer
dominates; it is mid-pack on Corr. Early fusion is the weakest GloVe row —
a shared LSTM on `409` features underperforms late concat.

Every GloVe fusion row is worse than the corresponding BERT row. The text
encoder upgrade is larger than the fusion-method gap.

## MOSEI — GMTM vs best fusion

Pulling the full-triple GMTM rows from the ablation CSVs:

| Text | Model | MAE | Acc2 | Corr | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| BERT | TransformerLate (sweep) | 0.5846 | 0.8393 | 0.7041 | 0.8699 |
| BERT | GMTM text+audio+visual | **0.5640** | **0.8429** | **0.7255** | **0.8777** |
| GloVe | LMF (sweep) | 0.6174 | **0.8107** | 0.6704 | **0.8523** |
| GloVe | GMTM text+audio+visual | **0.6251** | 0.8079 | **0.6714** | 0.8474 |

GMTM is a clear win on BERT. On GloVe it is essentially tied with LMF
(slightly worse MAE, slightly better Corr). If you only retrain one
personal model, retrain **BERT GMTM** or **BERT TransformerLate**.

Full leave-one-out tables: [ablation.md](ablation.md).

## MOSI — BERT (MOSEI checkpoints, merged MOSI splits)

Source: `model/mosi_test/mosi_bert_results.csv`.

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

TransformerLate transfers best among the fusion sweep. GMTM is mid-pack on
this merged MOSI BERT table; the ablation file shows **text+visual** GMTM
at MAE `0.9044`, close to TransformerLate.

The MOSI CSV uses the name `Concat` for late concat (script
`fusion_method == 'Concat'`).

## MOSI — GloVe (merged splits)

Source: `model/mosi_test/mosi_glove_results.csv`.

| Fusion | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GatedMultiTransfomer | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

Here GMTM **does** win every column. GloVe fusion models transfer poorly
(MAE `> 1.07`); the gated cross-modal stack is the only GloVe MOSI row
under `1.0` MAE.

## Cross-dataset pattern

```text
MOSEI BERT  >>  MOSEI GloVe  >>  MOSI BERT transfer  >>  MOSI GloVe transfer
```

MAE roughly doubles from the best MOSEI BERT model (`0.56`) to merged MOSI
GloVe fusion (`~1.08`), with GMTM GloVe MOSI in between (`0.97`).

Two modeling takeaways that stay inside this personal codebase:

1. **Upgrade text before fusion** — BERT vs GloVe is a bigger move than
   Concat vs LMF vs transformer on MOSEI.
2. **GMTM earns its keep when the text encoder is weaker or the target
   domain shifts** — GloVe MOSI is the clearest GMTM win; BERT MOSEI is a
   smaller GMTM win over an already-strong TransformerLate.

## Plotting

`model/results/plot.ipynb` is the original notebook for figures. The CSVs
are the source of truth if the notebook and the tables ever drift.

## How to add a new row later

1. Run the corresponding script with the same `single_test` path.
2. Append a line to the CSV **and** this page in the same commit.
3. State whether MOSI used the merged loader (default) or a real test split
   if you change `get_mosi_dataloader`.
