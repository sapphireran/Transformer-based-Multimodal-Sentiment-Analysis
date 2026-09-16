# File map

A guided tour of every tracked source file that is not a CSV.

## Root

| Path | Role |
| --- | --- |
| `README.md` | Project entry: layout, headline results, quick start |
| `LICENSE` | MIT, copyright 2024 pang990801 |
| `requirements.txt` | Runtime extras for training + examples |
| `.gitignore` | Drops pickles, checkpoints, caches, virtualenvs |
| `docs/` | Design and operator notes (this tree) |
| `examples/` | CPU demos that import `model/models.py` |

## `model/` — trainable code

| Path | Role |
| --- | --- |
| `models.py` | Encoders, fusion ops, custom Transformer, GMTM |
| `train_and_test.py` | `MultiFramework`, `train`, `single_test`, binning |
| `train_main_bert.py` | MOSEI BERT fusion sweep (train + test) |
| `train_main_glove.py` | MOSEI GloVe fusion sweep (test-only as checked in) |
| `train_GMTM_bert.py` | MOSEI BERT GMTM (test-only as checked in) |
| `train_GMTM_glove.py` | MOSEI GloVe GMTM ablation (test-only) |

## `model/data/` — loading

| Path | Role |
| --- | --- |
| `get_dataloader.py` | `Affectdataset`, packed / padded collate, ablation filters |
| `readme.md` | Where to put `.csd` and GloVe files |
| `glove.840B.300d.txt` | Placeholder; replace with the real GloVe dump |
| `analysis_dataset.ipynb` | Exploratory plots of a built pickle |
| `MOSEI/get_mosei.py` | SDK align + HDF5 sketch (MOSI field names inside) |
| `MOSEI/get_dataset.ipynb` | Interactive MOSEI pickle build |
| `MOSI/get_dataset.ipynb` | Interactive MOSI pickle build |

## `model/mosi_test/` — transfer

| Path | Role |
| --- | --- |
| `train_mosi_bert.py` | Score MOSEI BERT fusion ckpts on all-MOSI |
| `train_mosi_glove.py` | GloVe twin (broken MultiBench import) |
| `mult_bert_mosi.py` | GMTM BERT transfer |
| `mult_glove_mosi.py` | GMTM GloVe transfer + CSV write |
| `*.csv` | Last recorded MOSI numbers |

## `model/results/`

Snapshot CSVs plus `plot.ipynb` (large, image-heavy notebook that
renders the tables). Prefer the markdown tables in
[results.md](results.md) over re-opening the notebook.

## `model/checkpoints/`

Empty in git. `readme.md` and `ablation/readme.md` mark where
`torch.save` should write.

## `examples/`

| Path | Role |
| --- | --- |
| `synthetic_data.py` | MOSI/MOSEI-shaped random batches |
| `metrics.py` | Headless copy of the eval arithmetic |
| `run_fusion_demo.py` | Forward each fusion op |
| `run_gmtm_demo.py` | GMTM forward + optional unimodal zeros |
| `run_metrics_demo.py` | Score a synthetic `(y, ŷ)` pair |
| `run_tiny_train.py` | A few CPU AdamW steps on ConcatLate and GMTM |
| `compare_csv.py` | Diff a fresh sweep against `model/results/` |
| `shapes.py` | Named constants for 35 / 74 / 768 / 300 / 50 |
