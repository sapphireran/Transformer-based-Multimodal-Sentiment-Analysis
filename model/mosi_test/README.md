# MOSI transfer scripts

These scripts **do not train**. They load MOSEI checkpoints and score them
on MOSI features.

Important loader detail: `get_mosi_dataloader` / `get_ablation_mosi_dataloader`
concatenate MOSI train+valid+test into one eval set. The CSVs in this
folder are therefore “whole-pool transfer”, not the official MOSI test
split. Use `get_dataloader` if you need the real split.

## Scripts

| Script | Text | Checkpoints |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT | `../checkpoints/{Fusion}.pt` |
| `train_mosi_glove.py` | GloVe | `../checkpoints/glove_{Fusion}.pt` |
| `mult_bert_mosi.py` | BERT | `../checkpoints/ablation/model_{mods}.pt` |
| `mult_glove_mosi.py` | GloVe | `../checkpoints/ablation/model_glove_{mods}.pt` |

Run them from **this directory** so the relative pickle paths
(`../data/MOSI/mosi_raw_*.pkl`) resolve.

## Checked-in tables

| CSV | Contents |
| --- | --- |
| `mosi_bert_results.csv` | six fusions + GMTM, BERT |
| `mosi_glove_results.csv` | six fusions + GMTM, GloVe |
| `ablation_mosi_results.csv` | GMTM modality grid, BERT |
| `ablation_mosi_glove_results.csv` | GMTM modality grid, GloVe |

Commentary: [`../../docs/experiments-and-results.md`](../../docs/experiments-and-results.md).
