# MOSI transfer folder

These scripts load **MOSEI** checkpoints and score **MOSI** tensors.
They do not train on MOSI.

## Split warning (read this)

`get_mosi_dataloader` / `get_ablation_mosi_dataloader` concatenate MOSI
train + valid + test. Every CSV in this folder is in-corpus, not the
standard MOSI test split. Full discussion:
[`../../notes/04-mosi-transfer.md`](../../notes/04-mosi-transfer.md).

## Scripts

| Script | Front | What it loads | Writes |
| ------ | ----- | ------------- | ------ |
| `train_mosi_bert.py` | BERT | `../checkpoints/{method}.pt` | `mosi_bert_results.csv` |
| `train_mosi_glove.py` | GloVe | `../checkpoints/glove_{method}.pt` | `mosi_glove_results.csv` |
| `mult_bert_mosi.py` | BERT GMTM | `../checkpoints/ablation/model_{mods}.pt` | `ablation_mosi_results.csv` (write currently commented) |
| `mult_glove_mosi.py` | GloVe GMTM | `../checkpoints/ablation/model_glove_{mods}.pt` | `ablation_mosi_glove_results.csv` |

`train_mosi_glove.py` still imports MultiBench
`training_structures.Supervised_Learning`. Fix that import before a
clean re-run.

Method string `'Concat'` means the late-concat architecture but looks
for `Concat.pt` / `glove_Concat.pt`, not `ConcatLate.pt`. See
[`../checkpoints/readme.md`](../checkpoints/readme.md).

## Logged headline (merged MOSI)

BERT late transformer MAE **0.8986** (best of the six).
GloVe GMTM trimodal MAE **0.9748** (best of that table).
BERT GMTM T+V MAE **0.9044** beats BERT GMTM trimodal **0.9493**.

Do not put these next to a paper MOSI test number.
