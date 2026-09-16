# Published result tables

Copies of the CSVs written by the training / transfer scripts, kept
here so a working-directory overwrite of `model/*.csv` does not lose
the numbers. Narrative and ranking live in
[`docs/results.md`](../../docs/results.md). Reprint everything with:

```bash
python examples/inspect_results.py
```

| File | Experiment |
| --- | --- |
| `main_results.csv` | MOSEI BERT fusion bake-off |
| `glove_results.csv` | MOSEI GloVe fusion bake-off |
| `ablation_results.csv` | MOSEI BERT GMTM modalities |
| `ablation_glove_results.csv` | MOSEI GloVe GMTM modalities |

`plot.ipynb` is a local plotting scratchpad (large outputs). Prefer
the Markdown tables in `docs/results.md` when you just need the
numbers.
