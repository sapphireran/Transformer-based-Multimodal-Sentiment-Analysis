# Published results

Numbers below are the CSVs already in the repo, rounded to 4 decimals by
the training scripts. They are **not** reproduced in this documentation
pass. The plotter `examples/plot_published_results.py` redraws them.

Metric definitions: [evaluation.md](evaluation.md). Ablation commentary:
[ablation-study.md](ablation-study.md).

## MOSEI + BERT (fusion sweep)

[`model/results/main_results.csv`](../model/results/main_results.csv)

| Fusion | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |

Rank by MAE: TransformerLate < LMF < TFN < TransformerEarly < ConcatLate
< ConcatEarly.

Late transformer fusion is the only sweep method that clears 0.70
Pearson and 0.46 Acc-7. Early concat and late concat are nearly tied;
moving the LSTM after concat vs. before concat did not matter much once
BERT text is present.

GMTM (full) from the ablation table, for comparison on the same split:

| Model | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GMTM text+audio+visual | 0.5640 | 0.4827 | 0.5561 | 0.8429 | 0.7255 | 0.8777 |

That is a 0.02 MAE gap under TransformerLate, the previous best.

## MOSEI + GloVe (fusion sweep)

[`model/results/glove_results.csv`](../model/results/glove_results.csv)

| Fusion | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| LowRankTensorFusion | 0.6174 | 0.4458 | 0.5287 | 0.8107 | 0.6704 | 0.8523 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |

With 300-d word vectors the ranking changes: **LMF wins the sweep**
(MAE 0.6174). TransformerLate is third and no longer special. Early
fusion methods (concat / transformer) suffer more than late ones —
a wide BERT sequence concatenated with vision/audio is more informative
than a wide GloVe sequence in the same slot.

GMTM full (GloVe ablation): MAE **0.6251**, Acc-7 0.4467, Corr 0.6714.
That is slightly *behind* LMF on MAE, slightly *ahead* on Acc-7 / Corr.
GloVe is the setting where “use LMF” and “use GMTM” are a real trade.

## MOSI transfer + BERT

[`model/mosi_test/mosi_bert_results.csv`](../model/mosi_test/mosi_bert_results.csv)

MOSEI checkpoints, MOSI all-splits loader. ConcatLate is labeled
`Concat` in this file.

| Fusion | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat (late) | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GatedMultiTransfomer | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

Absolute errors jump by ~0.3 MAE versus MOSEI (domain shift + the MOSI
loader mixing train utterances into the test pool). TransformerLate
transfers best. Full GMTM is mid-pack; the ablation table shows
text+visual GMTM at 0.9044 MAE, close to TransformerLate.

## MOSI transfer + GloVe

[`model/mosi_test/mosi_glove_results.csv`](../model/mosi_test/mosi_glove_results.csv)

| Fusion | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat (late) | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| **GatedMultiTransfomer** | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

This is GMTM's best relative showing: it is the only GloVe-MOSI row under
1.0 MAE and the only one above 0.30 Acc-7. Late concat / LMF / late
transformer cluster around 1.08.

## Cross-setting sketch

```
                    MOSEI BERT   MOSEI GloVe   MOSI BERT   MOSI GloVe
Best sweep MAE      T-Late 0.58  LMF 0.62      T-Late 0.90 LMF/Concat ~1.08
GMTM full MAE       0.564        0.625         0.949       0.975
GMTM vs best sweep  wins         ~ties LMF     loses       wins
Text-only GMTM      0.569        0.662         0.936       1.009
```

Takeaways I would actually write on a slide:

1. **BERT + MOSEI:** spend capacity on cross-modal transformers (late
   fusion or GMTM). Concat is leaving ~0.05 MAE on the table.
2. **GloVe + MOSEI:** LMF is enough; GMTM is optional.
3. **Transfer to MOSI:** expect +0.3 to +0.5 MAE. Audio features are the
   first thing to distrust (negative Corr in the BERT MOSI ablation).
4. **Never report audio-only / vision-only as competitive** on these
   official COVAREP / FACET widths. They hover at 0.81–0.83 MAE on MOSEI
   and 1.37 on MOSI.

## Plotting

```bash
python examples/plot_published_results.py
# writes examples/output/mosei_mae_comparison.png
#        examples/output/mosei_ablation_mae.png
#        examples/output/mosi_transfer_mae.png
```

`examples/run_results_table.py` prints the same tables as GitHub-flavored
markdown, which is handy when you edit this file after a rerun.
