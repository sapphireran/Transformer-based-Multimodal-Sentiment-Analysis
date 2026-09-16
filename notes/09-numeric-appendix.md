# 09 — Numeric appendix

Every logged float in one place, copied from the CSVs, so I can grep
this file instead of opening four folders. If a CSV and this page
disagree later, the **CSV wins** and I patch this page.

Protocol reminders (do not skip):

- Acc-7 / Acc-5 = uniform bins on `[-3, 3]`, not rounded integers.
- MOSI rows = merged train+valid+test.
- GMTM subsets = zero-fill, still a 3-way module.
- One seed unless a note says otherwise.

## A. MOSEI BERT fusion (`main_results.csv`)

```text
method                    MAE     Acc7    Acc5    Acc2    Corr    F1
ConcatEarly               0.6178  0.4424  0.5221  0.8084  0.6652  0.8475
ConcatLate                0.6148  0.4508  0.5162  0.8084  0.6637  0.8460
LowRankTensorFusion       0.5970  0.4585  0.5409  0.8305  0.6925  0.8638
TensorFusion              0.6022  0.4445  0.5256  0.8244  0.6776  0.8626
TransformerEarly          0.6057  0.4514  0.5327  0.8206  0.6788  0.8604
TransformerLate           0.5846  0.4675  0.5460  0.8393  0.7041  0.8699
```

Best: TransformerLate, all columns.

## B. MOSEI GloVe fusion (`glove_results.csv`)

```text
method                    MAE     Acc7    Acc5    Acc2    Corr    F1
ConcatEarly               0.6782  0.4123  0.4919  0.7664  0.5942  0.8206
ConcatLate                0.6329  0.4373  0.5187  0.7878  0.6528  0.8325
LowRankTensorFusion       0.6174  0.4458  0.5287  0.8107  0.6704  0.8523
TensorFusion              0.6337  0.4336  0.5141  0.8038  0.6435  0.8457
TransformerEarly          0.6649  0.4207  0.5020  0.7821  0.6063  0.8373
TransformerLate           0.6287  0.4398  0.5076  0.7975  0.6397  0.8425
```

Best: LowRankTensorFusion, all columns.

## C. MOSEI GMTM BERT (`ablation_results.csv`)

```text
subset                    MAE     Acc7    Acc5    Acc2    Corr    F1
text                      0.5687  0.4728  0.5516  0.8404  0.7202  0.8741
audio                     0.8306  0.4127  0.4134  0.6252  0.1124  0.7552
visual                    0.8217  0.4012  0.4127  0.6293  0.2061  0.7693
text+audio                0.5659  0.4806  0.5596  0.8409  0.7227  0.8731
text+visual               0.5667  0.4838  0.5538  0.8324  0.7143  0.8697
audio+visual              0.8220  0.3951  0.4149  0.6274  0.2231  0.7670
text+audio+visual         0.5640  0.4827  0.5561  0.8429  0.7255  0.8777
```

Best MAE/Acc2/Corr/F1: T+A+V. Best Acc7: T+V (0.4838). Text-only is
within 0.005 MAE of the winner.

## D. MOSEI GMTM GloVe (`ablation_glove_results.csv`)

```text
subset                    MAE     Acc7    Acc5    Acc2    Corr    F1
text                      0.6616  0.4220  0.5091  0.8118  0.6557  0.8509
audio                     0.8232  0.4127  0.4134  0.6362  0.2067  0.7668
visual                    0.8139  0.3977  0.4185  0.6480  0.2256  0.7675
text+audio                0.7200  0.3503  0.4550  0.7760  0.5503  0.8186
text+visual               0.6612  0.4145  0.5207  0.7980  0.6459  0.8466
audio+visual              0.8141  0.3958  0.4205  0.6489  0.2345  0.7678
text+audio+visual         0.6251  0.4467  0.5317  0.8079  0.6714  0.8474
```

Best MAE/Acc7/Acc5/Corr: T+A+V. Best Acc2/F1: text-only.
Red row: text+audio.

## E. MOSI merged BERT fusion (`mosi_bert_results.csv`)

```text
method                    MAE     Acc7    Acc5    Acc2    Corr    F1
ConcatEarly               0.9448  0.2839  0.3574  0.7604  0.6284  0.7704
Concat                    1.0056  0.2757  0.3574  0.7339  0.5726  0.7287
LowRankTensorFusion       0.9525  0.3156  0.4063  0.7676  0.6149  0.7626
TensorFusion              0.9974  0.2616  0.3514  0.7705  0.6352  0.7733
TransformerEarly          0.9534  0.2868  0.3610  0.7681  0.6384  0.7782
TransformerLate           0.8986  0.3230  0.3894  0.7959  0.6748  0.7855
```

Best: TransformerLate, all columns except Acc5 (LMF 0.4063).

## F. MOSI merged GloVe fusion (`mosi_glove_results.csv`)

```text
method                    MAE     Acc7    Acc5    Acc2    Corr    F1
ConcatEarly               1.1702  0.1932  0.2458  0.6869  0.4774  0.6840
Concat                    1.0797  0.2267  0.2943  0.7064  0.5520  0.7139
LowRankTensorFusion       1.0825  0.2332  0.3046  0.7336  0.5559  0.7317
TensorFusion              1.1693  0.2158  0.2849  0.6737  0.4958  0.6418
TransformerEarly          1.1941  0.1951  0.2483  0.6938  0.4822  0.6909
TransformerLate           1.0855  0.2263  0.3097  0.7245  0.5303  0.7271
GatedMultiTransfomer      0.9748  0.3152  0.3995  0.7355  0.6177  0.7381
```

The GMTM row is from the ablation file (same floats) so I could rank
it with the six. Best: GMTM.

## G. MOSI merged GMTM BERT (`ablation_mosi_results.csv`)

```text
subset                    MAE     Acc7    Acc5    Acc2    Corr    F1
text                      0.9363  0.3170  0.4127  0.7667  0.6482  0.7559
audio                     1.3852  0.1681  0.1842  0.4509 -0.1256  0.4532
visual                    1.3741  0.1663  0.1837  0.5165 -0.0278  0.6800
text+audio                0.9349  0.3055  0.3995  0.7786  0.6494  0.7715
text+visual               0.9044  0.3257  0.4104  0.7796  0.6623  0.7767
audio+visual              1.3711  0.1654  0.1846  0.5208  0.0226  0.6807
text+audio+visual         0.9493  0.3078  0.3981  0.7638  0.6403  0.7626
```

Best: text+visual. Audio-only Corr negative.

## H. MOSI merged GMTM GloVe (`ablation_mosi_glove_results.csv`)

```text
subset                    MAE     Acc7    Acc5    Acc2    Corr    F1
text                      1.0085  0.3014  0.3903  0.7317  0.5959  0.7343
audio                     1.3799  0.1681  0.1842  0.4696 -0.0720  0.5513
visual                    1.3692  0.1667  0.1832  0.5036  0.0409  0.6177
text+audio                1.2863  0.1745  0.2400  0.6033  0.3524  0.5141
text+visual               1.0057  0.2739  0.3729  0.7235  0.5967  0.7446
audio+visual              1.3734  0.1672  0.1837  0.4998 -0.0042  0.6024
text+audio+visual         0.9748  0.3152  0.3995  0.7355  0.6177  0.7381
```

Best MAE: T+A+V. Red row: text+audio.

## Deltas I quote most often

```text
MOSEI BERT GMTM  text → trimodal     MAE  0.5687 → 0.5640   (−0.0047)
MOSEI GloVe GMTM text → trimodal     MAE  0.6616 → 0.6251   (−0.0365)
MOSEI GloVe GMTM text → text+audio   MAE  0.6616 → 0.7200   (+0.0584)
MOSEI BERT fusion lateTF vs LMF      MAE  0.5846 vs 0.5970  (−0.0124)
MOSEI GloVe fusion LMF vs lateTF     MAE  0.6174 vs 0.6287  (−0.0113)
MOSEI BERT lateTF → MOSI lateTF      MAE  0.5846 → 0.8986   (merged MOSI)
MOSI BERT GMTM T+V vs trimodal       MAE  0.9044 vs 0.9493  (−0.0449)
```

## Rankings in one line

```text
MOSEI BERT fusion : LateTF > LMF > TFN ≈ EarlyTF > concat
MOSEI GloVe fusion: LMF > LateTF ≈ LateConcat ≈ TFN > EarlyTF > EarlyConcat
MOSEI BERT GMTM   : TAV ≳ TA ≳ TV ≳ T  >>> V ≳ AV ≳ A
MOSEI GloVe GMTM  : TAV > T ≳ TV  >>> A ≳ AV ≳ V   (TA last)
MOSI BERT fusion  : LateTF > EarlyConcat ≈ LMF ≈ EarlyTF > TFN > Concat
MOSI GloVe fusion : GMTM > LMF ≈ Concat ≈ LateTF > TFN ≈ EarlyConcat > EarlyTF
MOSI BERT GMTM    : TV > TA ≳ T > TAV  >>> AV ≳ V ≳ A
MOSI GloVe GMTM   : TAV > TV ≳ T  >>> TA  >>> V ≳ AV ≳ A
```
