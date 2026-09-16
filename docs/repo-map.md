# Repository map

This is a personal research tree. Almost all experiment code sits under
`model/`. Documentation and CPU examples live at the repo root so they can be
read without installing the CMU SDK or downloading checkpoints.

## Root

| Path | What it is |
| --- | --- |
| `README.md` | Project overview and quick start |
| `LICENSE` | MIT (copyright 2024 pang990801) |
| `requirements.txt` | Training + example dependencies |
| `docs/` | Long-form notes |
| `examples/` | Runnable CPU walkthroughs |
| `tests/` | Shape / metric / collate checks |

## `model/` — experiment code

| File | What it does |
| --- | --- |
| `models.py` | Building blocks: LSTM/GRU/MLP, early/late concat, TFN, LMF, transformer encoders, `GatedMultiTransfomerModel` |
| `train_and_test.py` | `MultiFramework` (encoders → fusion → head), AdamW/RMSprop train loop, MOSI/MOSEI metric suite |
| `train_main_bert.py` | Six baseline fusion methods on MOSEI with BERT text |
| `train_main_glove.py` | Same sweep with GloVe text (hidden sizes scaled down) |
| `train_GMTM_bert.py` | GMTM on MOSEI BERT; modality list defaults to all three streams |
| `train_GMTM_glove.py` | GMTM GloVe ablation over the seven modality subsets |
| `main_results.csv` / `glove_results.csv` | Copies of the MOSEI baseline tables |
| `ablation_results.csv` / `ablation_glove_results.csv` | Copies of the GMTM ablation tables |

`train_and_test.test()` currently calls the inner test function twice (once
inside `all_in_one_test` for timing, once to return the dict). `single_test`
also opens a matplotlib confusion-matrix window. That is fine on a local GPU
box and noisy in headless CI — the examples never call `single_test`.

## `model/data/`

| Path | What it does |
| --- | --- |
| `get_dataloader.py` | `Affectdataset`, collate functions, MOSEI loaders, MOSI loaders, ablation zero-masking |
| `readme.md` | Expected on-disk names for GloVe and the `.csd` computational sequences |
| `MOSEI/get_mosei.py` | CMU-SDK alignment sketch (paths are machine-local) |
| `MOSEI/get_dataset.ipynb` | Notebook used to build `mosei_raw_bert.pkl` / `mosei_raw_glove.pkl` |
| `MOSI/get_dataset.ipynb` | Same for MOSI |
| `analysis_dataset.ipynb` | Peek at pickle keys and tensor shapes |

Pickle files themselves are **not** in git. They are large and come from the
CMU Multimodal SDK.

## `model/mosi_test/`

Scripts that load a checkpoint trained on MOSEI and score a MOSI pickle.

| File | Embedding | Task |
| --- | --- | --- |
| `train_mosi_bert.py` | BERT | Baseline transfer |
| `train_mosi_glove.py` | GloVe | Baseline transfer (imports an older `training_structures` path) |
| `mult_bert_mosi.py` | BERT | GMTM transfer |
| `mult_glove_mosi.py` | GloVe | GMTM transfer + writes `ablation_mosi_glove_results.csv` |
| `*_results.csv` | — | Recorded transfer numbers |

`get_mosi_dataloader` / `get_ablation_mosi_dataloader` concatenate the MOSI
train, valid, and test splits before scoring. Read [`data-pipeline.md`](data-pipeline.md)
before comparing those rows to a paper that uses the official MOSI test set only.

## `model/checkpoints/`

Empty placeholders plus short readmes. Expected names:

- Baselines: `ConcatEarly.pt`, `ConcatLate.pt`, `LowRankTensorFusion.pt`,
  `TensorFusion.pt`, `TransformerEarly.pt`, `TransformerLate.pt`
- GloVe baselines: same names with a `glove_` prefix
- GMTM ablation: `ablation/model_{text+audio+visual}.pt` (BERT) or
  `ablation/model_glove_{...}.pt`

## `model/results/`

Canonical copies of the CSVs plus `plot.ipynb` (bar / radar charts from a
local Windows run). The docs tables are transcribed from these files, not
re-fit in CI.

## `examples/` and `tests/`

Examples import `model/` by putting that directory on `sys.path`, matching the
original scripts. Tests reuse the same helpers so a broken shape change in
`models.py` fails in `pytest` instead of only in a GPU training job.
