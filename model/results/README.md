# Result CSVs

Canonical copies of personal experiment tables. Narrative:
[`../../docs/results.md`](../../docs/results.md) and
[`../../docs/ablation.md`](../../docs/ablation.md).

| File | Dataset | Text | What |
| --- | --- | --- | --- |
| `main_results.csv` | MOSEI | BERT | six fusion methods |
| `glove_results.csv` | MOSEI | GloVe | six fusion methods |
| `ablation_results.csv` | MOSEI | BERT | GMTM modality subsets |
| `ablation_glove_results.csv` | MOSEI | GloVe | GMTM modality subsets |

MOSI transfer CSVs live in `../mosi_test/` (`mosi_bert_results.csv`,
`mosi_glove_results.csv`, `ablation_mosi_*.csv`). Those MOSI numbers use a
**merged** train+valid+test loader.

`plot.ipynb` is the original figure notebook. Prefer the CSVs if the
notebook output is stale.

Duplicates of some CSVs also sit in `model/` (`main_results.csv`, …). If
they ever diverge, treat **this folder** as the copy described in the docs.
