# Result snapshots

These CSVs are the copy of record for the markdown tables in
[docs/results.md](../../docs/results.md). Re-running a training
script writes a CSV *next to the script* (`model/main_results.csv`,
`model/mosi_test/mosi_bert_results.csv`, …). Copy the new file here
if you want the snapshot — and the docs — to update together.

| File | Experiment |
| --- | --- |
| `main_results.csv` | MOSEI BERT, six fusion baselines |
| `glove_results.csv` | MOSEI GloVe, six fusion baselines |
| `ablation_results.csv` | MOSEI BERT GMTM, 7 modality subsets |
| `ablation_glove_results.csv` | MOSEI GloVe GMTM, 7 modality subsets |
| `plot.ipynb` | Charts of the tables above (large notebook) |

MOSI transfer CSVs live under `../mosi_test/`. Diff a fresh file
against a snapshot with:

```bash
python examples/compare_csv.py \
    --snapshot model/results/main_results.csv \
    --fresh model/main_results.csv
```
