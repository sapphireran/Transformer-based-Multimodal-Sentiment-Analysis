# MOSI transfer scripts

These files **do not train on MOSI**. They load MOSEI checkpoints from
`../checkpoints/` and score MOSI pickles.

| Script | Text | What it scores |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT | fusion-sweep checkpoints |
| `train_mosi_glove.py` | GloVe | fusion-sweep checkpoints |
| `mult_bert_mosi.py` | BERT | GMTM ablation checkpoints |
| `mult_glove_mosi.py` | GloVe | GMTM ablation checkpoints |

Loaders come from `get_mosi_dataloader` / `get_ablation_mosi_dataloader`.
Both **concatenate MOSI train + valid + test**. Do not treat the CSVs as
official MOSI test-split numbers.

| CSV | Matches |
| --- | --- |
| `mosi_bert_results.csv` | BERT fusion + GMTM row |
| `mosi_glove_results.csv` | GloVe fusion + GMTM row |
| `ablation_mosi_results.csv` | BERT GMTM subsets |
| `ablation_mosi_glove_results.csv` | GloVe GMTM subsets |

`train_mosi_glove.py` still imports `training_structures.Supervised_Learning`,
which is not in this repository. Switch that import to `train_and_test`
before running it.

Narrative: [`../../docs/results.md`](../../docs/results.md),
[`../../docs/ablation.md`](../../docs/ablation.md).
