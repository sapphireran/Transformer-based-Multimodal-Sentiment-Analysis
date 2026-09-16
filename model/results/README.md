# Recorded personal result tables

These CSVs are the source of the numbers in
[`../../docs/experiments-and-results.md`](../../docs/experiments-and-results.md)
and the root README.

| File | Split | Text | Sweep |
| --- | --- | --- | --- |
| `main_results.csv` | MOSEI test | BERT | six fusion baselines |
| `glove_results.csv` | MOSEI test | GloVe | six fusion baselines |
| `ablation_results.csv` | MOSEI test | BERT | GMTM × modality subset |
| `ablation_glove_results.csv` | MOSEI test | GloVe | GMTM × modality subset |
| `plot.ipynb` | — | — | personal plots |

Column order is always

```
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Acc-7 / Acc-5 are **uniform** bins on [-3, 3], not integer MOSI labels.

Copies of the same four tables also sit in `model/` next to the training
scripts (the scripts write `./main_results.csv` when launched from
`model/`). MOSI transfer tables live in `model/mosi_test/`.

Do not edit a CSV to “clean” a number without rerunning the corresponding
script — the docs quote these files verbatim.
