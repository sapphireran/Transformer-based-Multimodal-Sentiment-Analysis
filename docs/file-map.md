# File map

Personal index of every tracked path that is not a git hook.

## Root

| Path | Notes |
| --- | --- |
| `README.md` | Project entry, headline table, example commands |
| `LICENSE` | MIT, copyright 2024 pang990801 |
| `requirements.txt` | Python deps for train + examples |
| `.gitignore` | Drops weights, pickles, caches |

## `docs/`

| Path | Notes |
| --- | --- |
| `architecture.md` | Modules, shapes, GMTM forward |
| `data-pipeline.md` | Pickles, collate, ablation zeros, MOSI merge |
| `evaluation.md` | MAE / Acc7 / Acc2 definitions |
| `experiments.md` | All recorded CSV tables |
| `hyperparameters.md` | Optimizer, HParams, encoder widths |
| `reproducing.md` | Machine-to-machine checklist |
| `glossary.md` | MOSI / Facet / packed / etc. |
| `file-map.md` | This file |

## `examples/`

| Path | Notes |
| --- | --- |
| `README.md` | How to run each script |
| `synthetic_data.py` | Random clips with MOSI-like ranks |
| `fusion_forward.py` | One forward pass per fusion module |
| `gmtm_forward.py` | Shape printout through GMTM |
| `gmtm_toy_train.py` | 25-step L1 train on synthetic labels |
| `evaluate_metrics.py` | Acc7 / Acc5 / Acc2 on known scores |
| `packed_vs_padded.py` | Mimic `_process_1` vs `_process_2` |

## `tests/`

| Path | Notes |
| --- | --- |
| `test_synthetic_data.py` | Batch ranks and label range |
| `test_metrics.py` | Bin edges and exclude-zero F1 |
| `test_fusion_forward.py` | CPU forward shapes |
| `test_gmtm.py` | GMTM output `(B, 1)` and a train step |

## `model/` — training

| Path | Notes |
| --- | --- |
| `models.py` | All nn.Modules |
| `train_and_test.py` | `MultiFramework`, `train`, `test`, metrics |
| `train_main_bert.py` | MOSEI BERT fusion sweep |
| `train_main_glove.py` | MOSEI GloVe fusion sweep |
| `train_GMTM_bert.py` | MOSEI GMTM BERT ablation |
| `train_GMTM_glove.py` | MOSEI GMTM GloVe ablation |
| `main_results.csv` | Copy of BERT fusion table |
| `glove_results.csv` | Copy of GloVe fusion table |
| `ablation_results.csv` | Copy of BERT GMTM ablation |
| `ablation_glove_results.csv` | Copy of GloVe GMTM ablation |

## `model/data/`

| Path | Notes |
| --- | --- |
| `readme.md` | Where raw `.csd` / GloVe files live |
| `get_dataloader.py` | `Affectdataset`, collate, ablation filters |
| `analysis_dataset.ipynb` | Exploratory notebook |
| `MOSEI/get_mosei.py` | CMU-SDK align → HDF5 sketch |
| `MOSEI/get_dataset.ipynb` | Interactive MOSEI build |
| `MOSI/get_dataset.ipynb` | Interactive MOSI build |

## `model/mosi_test/`

| Path | Notes |
| --- | --- |
| `train_mosi_bert.py` | Eval MOSEI BERT checkpoints on MOSI |
| `train_mosi_glove.py` | Same for GloVe; still imports an old `training_structures` path |
| `mult_bert_mosi.py` | GMTM BERT on MOSI |
| `mult_glove_mosi.py` | GMTM GloVe on MOSI |
| `*.csv` | Transfer tables |

## `model/results/`

| Path | Notes |
| --- | --- |
| `main_results.csv` | BERT fusion (canonical copy) |
| `glove_results.csv` | GloVe fusion |
| `ablation_results.csv` | BERT GMTM ablation |
| `ablation_glove_results.csv` | GloVe GMTM ablation |
| `plot.ipynb` | Figures from the tables |

## `model/checkpoints/`

Only readmes are committed. Weights stay local.

| Path | Notes |
| --- | --- |
| `readme.md` | Main-experiment `.pt` directory |
| `ablation/readme.md` | Ablation `.pt` directory |
