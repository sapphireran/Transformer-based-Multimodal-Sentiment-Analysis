# Repository map

Personal index of every tracked file that is not a git hook. Paths are from the repo root.

## Top level

| Path | Role |
| --- | --- |
| `README.md` | Project entry, headline table, quick start |
| `LICENSE` | MIT, copyright pang990801 (2024) |
| `requirements.txt` | Train / example / test Python deps |
| `.gitignore` | Drops checkpoints, pickles, `.csd`, example output |
| `docs/` | Long-form notes (this tree) |
| `examples/` | CPU-only demos |
| `tests/` | Pytest wrappers around the demos |
| `model/` | All research code |

## `model/` — models and loops

| Path | Role |
| --- | --- |
| `models.py` | Identity, LSTM/GRU/MLP, fusion blocks, GMTM, sinusoidal attention stack |
| `train_and_test.py` | `MultiFramework`, `train`, `single_test`, `test`, Acc-5/7 helpers |
| `train_main_bert.py` | MOSEI BERT fusion sweep (currently 1 epoch) |
| `train_main_glove.py` | MOSEI GloVe fusion sweep (eval-only as committed) |
| `train_GMTM_bert.py` | MOSEI BERT GMTM / ablation (eval-only as committed) |
| `train_GMTM_glove.py` | MOSEI GloVe GMTM / ablation (eval-only as committed) |

## `model/data/` — loaders and builders

| Path | Role |
| --- | --- |
| `readme.md` | Where GloVe + `.csd` files live |
| `get_dataloader.py` | `Affectdataset`, official / ablation / MOSI-merge loaders, collate |
| `analysis_dataset.ipynb` | Personal EDA notebook |
| `MOSEI/get_mosei.py` | CMU-SDK align + HDF5 dump (MOSI paths despite the folder name) |
| `MOSEI/get_dataset.ipynb` | Interactive MOSEI pickle builder |
| `MOSI/get_dataset.ipynb` | Interactive MOSI pickle builder |

Expected but untracked inputs: `MOSEI/mosei_raw_{bert,glove}.pkl`, `MOSI/mosi_raw_{bert,glove}.pkl`, `glove.840B.300d.txt`, `*/cmu*/**.csd`.

## `model/mosi_test/` — transfer evaluation

| Path | Role |
| --- | --- |
| `train_mosi_bert.py` | Score MOSEI BERT fusion ckpts on merged MOSI |
| `train_mosi_glove.py` | GloVe twin; still imports MultiBench |
| `mult_bert_mosi.py` | Score BERT GMTM on merged MOSI |
| `mult_glove_mosi.py` | Score GloVe GMTM on merged MOSI |
| `mosi_bert_results.csv` | Published BERT transfer table |
| `mosi_glove_results.csv` | Published GloVe transfer table |
| `ablation_mosi_results.csv` | BERT GMTM transfer ablation |
| `ablation_mosi_glove_results.csv` | GloVe GMTM transfer ablation |

## `model/results/` — in-domain tables

| Path | Role |
| --- | --- |
| `main_results.csv` | MOSEI BERT fusion sweep |
| `glove_results.csv` | MOSEI GloVe fusion sweep |
| `ablation_results.csv` | MOSEI BERT GMTM ablation |
| `ablation_glove_results.csv` | MOSEI GloVe GMTM ablation |
| `plot.ipynb` | Personal bar / comparison notebook |

Duplicate CSVs also sit in `model/*.csv` (copies of the results folder). Prefer `model/results/` as the source of truth.

## `model/checkpoints/`

| Path | Role |
| --- | --- |
| `readme.md` | “main experiment `.pt` files go here” |
| `ablation/readme.md` | “ablation `.pt` files go here” |

The `.pt` files themselves are gitignored. Names the scripts look for are listed in [training](training.md).

## `docs/`

| Path | Role |
| --- | --- |
| `architecture.md` | Encoders, fusions, GMTM diagram, hyperparams |
| `datasets.md` | MOSI / MOSEI, pickle contract, loaders |
| `training.md` | Script matrix, recipes, packed vs padded |
| `evaluation.md` | MAE / Corr / Acc-2/5/7 / F1, MOSI merge |
| `results.md` | All eight CSV tables + takeaways |
| `reproducing.md` | End-to-end personal checklist |
| `repository_map.md` | This file |

## `examples/`

| Path | Role |
| --- | --- |
| `README.md` | How to run the demos |
| `common.py` | `sys.path` + synthetic `[B,T,D]` batches |
| `fusion_forward.py` | Forward pass of every fusion used in the sweep |
| `gmtm_forward.py` | GMTM BERT + GloVe + a zeroed-modality ablation |
| `metrics_demo.py` | Hand-built y / ŷ through the real metric helpers |
| `summarize_results.py` | Reprint CSVs as markdown |
| `run_all_examples.py` | One entry point |

## `tests/`

| Path | Role |
| --- | --- |
| `test_examples.py` | Shape / finite-output / CSV / metric assertions |

## Import graph (runtime)

```text
train_main_bert.py
    ├─ train_and_test.{train,test}
    ├─ models.{Concat*,Tensor*,*Transformer*,LSTM,GRU*,MLP,Identity}
    └─ data.get_dataloader.get_dataloader

train_GMTM_bert.py
    ├─ train_and_test.{train,test}
    ├─ models.GatedMultiTransfomerModel
    └─ data.get_dataloader.get_ablation_dataloader

mosi_test/train_mosi_bert.py
    ├─ train_and_test.test
    ├─ models.*
    └─ data.get_dataloader.get_mosi_dataloader
```

`models.py` does not import the loaders. `get_dataloader.py` does not import the models. That split is what lets the examples import fusion modules without touching a pickle.
