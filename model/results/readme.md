# Results copies

Duplicates of the four MOSEI CSVs that also live in `model/`, plus
`plot.ipynb` (large because it embeds figures).

| File | Same bytes as | Note |
| ---- | ------------- | ---- |
| `main_results.csv` | `../main_results.csv` | MOSEI BERT fusion |
| `glove_results.csv` | `../glove_results.csv` | MOSEI GloVe fusion |
| `ablation_results.csv` | `../ablation_results.csv` | GMTM BERT subsets |
| `ablation_glove_results.csv` | `../ablation_glove_results.csv` | GMTM GloVe subsets |

MOSI CSVs were **not** duplicated here; they stay in `../mosi_test/`.

I have not cleaned `plot.ipynb` outputs. If I regenerate plots, I should
strip the embedded images or move figures to a `notes/figures/` folder
that I choose to keep.

How to read the columns: [`../../docs/metrics.md`](../../docs/metrics.md).
Narrative: [`../../notes/00-lab-index.md`](../../notes/00-lab-index.md).
