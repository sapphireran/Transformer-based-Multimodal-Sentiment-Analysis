# Logged personal runs

CSV copies of the original MOSEI sweeps. `plot.ipynb` is the large
plotting notebook from that session (embedded figures). Prefer the CSVs.

| File | What |
| --- | --- |
| `main_results.csv` | BERT fusion sweep |
| `glove_results.csv` | GloVe fusion sweep |
| `ablation_results.csv` | BERT GMTM modality ablation |
| `ablation_glove_results.csv` | GloVe GMTM modality ablation |

How to read the columns: [`docs/metrics.md`](../../docs/metrics.md).
How to read the ranking: [`docs/experiments.md`](../../docs/experiments.md).

Pretty-print without torch:

```bash
python examples/print_logged_results.py
```

Do not overwrite these files on a new train. Add `*_rerun.csv` next to
them and leave a note in `docs/reproduction.md`.
