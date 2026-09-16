# Script reference

Working directory and files for every entry point. Run experiment scripts
from the directory in the **CWD** column so relative `data/` and
`checkpoints/` paths resolve.

## Documentation and examples (repo root)

| Command | Needs pickles / GPU | What it does |
| --- | --- | --- |
| `python examples/run_all.py` | no | every CPU walkthrough |
| `python examples/test_examples.py` | no | unittest wrapper |
| `python examples/synthetic_data.py` | no | print BERT/GloVe toy shapes |
| `python examples/packed_vs_padded.py` | no | show both collate nestings |
| `python examples/encoder_shapes.py` | no | LSTM / GRU / transformer I/O |
| `python examples/fusion_walkthrough.py` | no | concat / TFN / LMF / transformers |
| `python examples/metrics_demo.py` | no | Acc5/7/2 on a 6-row toy |
| `python examples/result_tables.py` | no | star the best cell in each CSV |
| `python examples/gmtm_toy_train.py` | no | short GMTM L1 overfit |

Install: `python -m pip install -r requirements.txt`.

## MOSEI training (`cd model`)

| Script | CWD | Reads | Writes |
| --- | --- | --- | --- |
| `train_main_bert.py` | `model/` | `data/MOSEI/mosei_raw_bert.pkl` | `checkpoints/{Method}.pt`, `main_results.csv` |
| `train_main_glove.py` | `model/` | `data/MOSEI/mosei_raw_glove.pkl` | `checkpoints/glove_{Method}.pt`, `glove_results.csv` |
| `train_GMTM_bert.py` | `model/` | BERT pickle | `checkpoints/ablation/model_{a+b}.pt` |
| `train_GMTM_glove.py` | `model/` | GloVe pickle | `checkpoints/ablation/model_glove_{a+b}.pt`, `ablation_glove_results.csv` |

Checked-in caveats (do not “fix” silently if you are comparing to the CSVs):

- `train_main_bert.py` uses `total_epochs=1`.
- GloVe fusion and both GMTM scripts have `train(...)` commented; they
  `torch.load` an existing `.pt`.
- Every script calls `.cuda()` at construction time.

## MOSI transfer (`cd model/mosi_test`)

| Script | CWD | Checkpoint glob | MOSI pickle |
| --- | --- | --- | --- |
| `train_mosi_bert.py` | `model/mosi_test/` | `../checkpoints/{Method}.pt` | `../data/MOSI/mosi_raw_bert.pkl` |
| `train_mosi_glove.py` | `model/mosi_test/` | `../checkpoints/glove_{Method}.pt` | `../data/MOSI/mosi_raw_glove.pkl` |
| `mult_bert_mosi.py` | `model/mosi_test/` | `../checkpoints/ablation/model_*.pt` | BERT MOSI |
| `mult_glove_mosi.py` | `model/mosi_test/` | `../checkpoints/ablation/model_glove_*.pt` | GloVe MOSI |

`train_mosi_glove.py` imports `training_structures.Supervised_Learning`
(missing from this repo). Point it at `train_and_test` first.

Loaders merge MOSI train+valid+test. See [datasets.md](datasets.md).

## Data builders

| Path | CWD | Role |
| --- | --- | --- |
| `model/data/MOSEI/get_dataset.ipynb` | notebook dir | build `mosei_raw_{bert,glove}.pkl` |
| `model/data/MOSI/get_dataset.ipynb` | notebook dir | build `mosi_raw_{bert,glove}.pkl` |
| `model/data/MOSEI/get_mosei.py` | edit paths first | MMSDK align + HDF5 dump |
| `model/data/analysis_dataset.ipynb` | notebook dir | exploratory plots |
| `model/results/plot.ipynb` | `model/results/` | figures from CSVs |

## Shared library modules (imported, not launched)

| Module | Symbols |
| --- | --- |
| `model/models.py` | encoders, fusion, GMTM |
| `model/train_and_test.py` | `MultiFramework`, `train`, `test`, `single_test` |
| `model/data/get_dataloader.py` | `get_dataloader`, ablation + MOSI variants |

## Environment variables

None are read. Device is `cuda:0` if `torch.cuda.is_available()` else CPU
inside `train_and_test.py`. Example scripts use `examples/common.py:device()`.

## Typical personal session

```bash
# 1. Prove the tree still imports and the CSVs parse.
python examples/test_examples.py
python examples/run_all.py

# 2. After pickles exist:
cd model
python train_main_bert.py          # bump epochs first
python train_GMTM_bert.py          # uncomment train() first

# 3. Optional transfer:
cd mosi_test
python train_mosi_bert.py
```
