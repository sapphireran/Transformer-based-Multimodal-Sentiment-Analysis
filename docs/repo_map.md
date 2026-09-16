# Repository map

Every training or evaluation entry point lives under `model/`. Documentation
and synthetic examples live at the repository root so they stay easy to find
without mixing them into experiment scripts.

## Top level

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and quick start |
| `requirements.txt` | Python packages for training scripts and examples |
| `LICENSE` | MIT |
| `docs/` | Long-form notes |
| `examples/` | CPU walkthroughs on synthetic tensors |
| `model/` | Original experiment code and recorded CSVs |

## `model/` experiment code

| File | What it does |
| --- | --- |
| `models.py` | Building blocks: LSTM / GRU / MLP, concat and tensor fusion, early/late transformers, `GatedMultiTransfomerModel`, positional encodings |
| `train_and_test.py` | `MultiFramework`, `train()`, `single_test()`, `eval_affect()`, Acc5/Acc7 binning |
| `train_main_bert.py` | MOSEI fusion sweep with BERT text (`35 / 74 / 768`) |
| `train_main_glove.py` | Same sweep with GloVe text (`35 / 74 / 300`) |
| `train_GMTM_bert.py` | GMTM on MOSEI BERT; optional modality list |
| `train_GMTM_glove.py` | GMTM on MOSEI GloVe; full 7-combination ablation |
| `main_results.csv`, `glove_results.csv`, `ablation_*.csv` | Copies of tables also stored under `results/` |

`train_main_glove.py` and several GMTM / MOSI scripts currently **load** a
checkpoint and skip `train(...)` (the train call is commented out). The BERT
fusion sweep still calls `train(...)` but only for `total_epochs=1` in the
checked-in file. Treat the scripts as experiment notebooks in `.py` form, not
as a frozen CLI.

## `model/data/`

| File | What it does |
| --- | --- |
| `get_dataloader.py` | `Affectdataset`, packed (`_process_1`) and padded (`_process_2`) collates, ablation zeroing |
| `MOSEI/get_mosei.py` | Example MMSDK alignment / HDF5 export (paths are machine-local) |
| `MOSEI/get_dataset.ipynb`, `MOSI/get_dataset.ipynb` | Pickle builders |
| `analysis_dataset.ipynb` | Exploratory plots |

Expected pickle names (not committed):

- `model/data/MOSEI/mosei_raw_bert.pkl`
- `model/data/MOSEI/mosei_raw_glove.pkl`
- `model/data/MOSI/mosi_raw_bert.pkl`
- `model/data/MOSI/mosi_raw_glove.pkl`

## `model/mosi_test/`

Scripts that load **MOSEI-trained** checkpoints and score them on MOSI. This
is a transfer / robustness check, not a from-scratch MOSI training loop.

| File | Embedding | Task |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT | fusion sweep on MOSI |
| `train_mosi_glove.py` | GloVe | fusion sweep on MOSI |
| `mult_bert_mosi.py` | BERT | GMTM on MOSI |
| `mult_glove_mosi.py` | GloVe | GMTM on MOSI |
| `*_results.csv` | — | recorded MOSI scores |

`get_mosi_dataloader` concatenates MOSI train+valid+test into one evaluation
loader. Do not compare those numbers directly to papers that score only the
official MOSI test split.

## `model/checkpoints/`

Place `.pt` files here. The training scripts save:

- `checkpoints/{FusionMethod}.pt` — BERT fusion sweep
- `checkpoints/glove_{FusionMethod}.pt` — GloVe fusion sweep
- `checkpoints/ablation/model_{text+audio+visual}.pt` — GMTM BERT
- `checkpoints/ablation/model_glove_{...}.pt` — GMTM GloVe

Nothing under this directory is committed except short readmes.

## `model/results/`

Canonical copies of the CSV tables, plus `plot.ipynb` for figures.

## Import path habits

Most scripts do:

```python
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
```

and expect you to **run them from `model/`** (or `model/mosi_test/`). The
`examples/` scripts instead add `model/` to `sys.path` so they can be launched
from the repository root.

`train_mosi_glove.py` still imports
`training_structures.Supervised_Learning`. That module is **not** in this
repository; use `train_and_test` the way the BERT MOSI script does if you
revive that file.
