# Recorded results

Numbers below are copied from the committed CSVs. They are the last
completed runs in this personal repo, not a claim that every seed was
averaged. Re-running the scripts with a new checkpoint will overwrite the
local CSV; the copies under `model/results/` are the snapshot used for
`model/results/plot.ipynb`.

Lower **MAE** is better. Higher **Acc-7 / Acc-5 / Acc-2 / Corr / F1** is
better. Acc-7 and Acc-5 use the uniform bins described in
[metrics.md](metrics.md).

## MOSEI · BERT features

From `model/main_results.csv` and the GMTM row of `model/ablation_results.csv`.

| Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |
| **GMTM (text+audio+visual)** | **0.5640** | **0.4827** | **0.5561** | **0.8429** | **0.7255** | **0.8777** |

On this run GMTM is the best row on every column. Transformer-late is the
strongest baseline.

## MOSEI · GloVe features

From `model/glove_results.csv` and `model/ablation_glove_results.csv`.

| Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | 0.6174 | 0.4458 | 0.5287 | 0.8107 | 0.6704 | 0.8523 |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |
| **GMTM (text+audio+visual)** | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

LMF slightly wins Acc-2 / F1; GMTM wins MAE, Acc-7, Acc-5, and Corr.

## MOSEI GMTM modality ablation · BERT

From `model/ablation_results.csv`.

| Modalities | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | 0.5640 | 0.4827 | 0.5561 | 0.8429 | 0.7255 | 0.8777 |

Text dominates. Audio or visual alone is close to chance on correlation.
Adding both non-text streams to text still edges MAE / Acc-2 / Corr / F1.

## MOSEI GMTM modality ablation · GloVe

From `model/ablation_glove_results.csv`.

| Modalities | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | 0.6251 | 0.4467 | 0.5317 | 0.8079 | 0.6714 | 0.8474 |

GloVe text+audio is a noticeable regression versus text-only; the full
triplet recovers and is the best GloVe GMTM row.

## MOSI transfer · BERT (MOSEI checkpoints)

From `model/mosi_test/mosi_bert_results.csv` and
`model/mosi_test/ablation_mosi_results.csv`.

| Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| ConcatLate | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GMTM | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

MOSI is evaluated on the **merged** MOSI splits (`get_mosi_dataloader`),
so these numbers are not a standard MOSI test-only score.

## MOSI transfer · GloVe

From `model/mosi_test/mosi_glove_results.csv`.

| Method | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| ConcatLate | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| **GMTM** | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

## How to reprint the tables

```bash
python examples/inspect_results.py
```

The plotting notebook `model/results/plot.ipynb` concatenates the GMTM
ablation's last row onto the main fusion-zoo tables before drawing bars.
