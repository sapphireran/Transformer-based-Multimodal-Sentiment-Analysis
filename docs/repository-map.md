# Repository map

Everything that trains or evaluates a model lives under `model/`. Documentation
and CPU examples were added at the repo root so they stay out of the original
training working directory.

## Root

| File | Role |
| --- | --- |
| `README.md` | Project entry, headline table, commands |
| `LICENSE` | MIT, copyright 2024 pang990801 |
| `requirements.txt` | Training extras (`memory-profiler`, CUDA torch, etc.) |
| `requirements-examples.txt` | CPU demo / pytest stack |
| `docs/` | The notes you are reading |
| `examples/` | Synthetic data, forward passes, toy GMTM, plots |

## `model/models.py`

Single module, several families of `nn.Module`:

- **Pass-through / MLP / RNN:** `Identity`, `MLP`, `Linear` (later shadowed by
  a factory function of the same name), `LSTM`, `GRU`, `GRUWithLinear`.
- **Unimodal transformers:** `Transformer` (last timestep only),
  `TransformerSeq` (full sequence).
- **Fusion:** `ConcatEarly`, `ConcatLate`, `TensorFusion`,
  `LowRankTensorFusion`, `TransformerFusion`, `EarlyFusionTransformer`,
  `LateFusionTransformer`.
- **Custom transformer stack** used by GMTM: `TransformerEncoder`,
  `TransformerEncoderLayer`, `SinusoidalPositionalEmbedding`,
  `AttentionPooling`.
- **GMTM:** `GatedMultiTransfomerModel` (the name keeps the original
  spelling).

The file is imported by every `train_*.py` and by `examples/run_*.py`.

## `model/train_and_test.py`

- `MultiFramework` — wraps `(encoders, fusion, head)`.
- `deal_with_objective` — `CrossEntropyLoss` vs `L1Loss` / `MSELoss` /
  `BCEWithLogitsLoss`.
- `train` — AdamW-capable loop, grad clip, best-`pt` on validation L1,
  optional `memory_profiler` wrapping.
- `single_test` / `test` — MAE, MSE, Pearson r, Acc-7, Acc-5, Acc-2, F1.
  Also opens a matplotlib confusion matrix (`plt.show()`), which is why the
  examples reimplement the metrics instead of calling `test()` headless.
- `split_uniform_7`, `split_uniform_5`, `eval_affect` — the binning helpers.

## `model/data/`

| File | Role |
| --- | --- |
| `get_dataloader.py` | `Affectdataset`, `get_dataloader`, MOSI-only loaders, ablation filters, `_process_1` / `_process_2` |
| `readme.md` | Expected `.csd` / GloVe filenames |
| `MOSEI/get_mosei.py` | MMSDK alignment + HDF5 dump (paths are local Windows paths from the original machine) |
| `MOSEI/get_dataset.ipynb` | Interactive rebuild of `mosei_raw_*.pkl` |
| `MOSI/get_dataset.ipynb` | Same for MOSI |
| `analysis_dataset.ipynb` | Peek at pickle keys and array shapes |

`get_dataloader.py` is the only data file the training scripts import.

## Training entry points

All of these expect the current working directory to be `model/` (or
`model/mosi_test/` for the transfer scripts).

| Script | Embedding | Dataset | What it sweeps |
| --- | --- | --- | --- |
| `train_main_bert.py` | BERT | MOSEI | 6 fusion methods, trains + tests |
| `train_main_glove.py` | GloVe | MOSEI | 6 fusion methods (train call is commented; loads `glove_*.pt`) |
| `train_GMTM_bert.py` | BERT | MOSEI | GMTM; train call commented; ablation combos mostly commented |
| `train_GMTM_glove.py` | GloVe | MOSEI | GMTM over the 7 modality subsets |
| `mosi_test/train_mosi_bert.py` | BERT | MOSI | Load MOSEI fusion checkpoints, test |
| `mosi_test/train_mosi_glove.py` | GloVe | MOSI | Same (still imports MultiBench `training_structures` in one place) |
| `mosi_test/mult_bert_mosi.py` | BERT | MOSI | GMTM checkpoint on MOSI |
| `mosi_test/mult_glove_mosi.py` | GloVe | MOSI | GMTM checkpoint on MOSI |

Checkpoint directories:

- `model/checkpoints/` — `ConcatEarly.pt`, `TransformerLate.pt`,
  `glove_ConcatLate.pt`, …
- `model/checkpoints/ablation/` — `model_text+audio+visual.pt`,
  `model_glove_text.pt`, …

Those `.pt` files are gitignored. The CSVs that record their scores are not.

## Result CSVs (duplicated on purpose)

The same tables appear both at `model/*.csv` and `model/results/*.csv`, and
again under `model/mosi_test/` for the transfer runs. `examples/results_lib.py`
prefers `model/results/` and `model/mosi_test/` so plots stay stable if you
re-run a training script and overwrite the copies in `model/`.

| CSV | Setting |
| --- | --- |
| `results/main_results.csv` | MOSEI + BERT, 6 fusions |
| `results/glove_results.csv` | MOSEI + GloVe, 6 fusions |
| `results/ablation_results.csv` | MOSEI + BERT, GMTM modality subsets |
| `results/ablation_glove_results.csv` | MOSEI + GloVe, GMTM modality subsets |
| `mosi_test/mosi_bert_results.csv` | MOSI transfer, BERT fusions + GMTM |
| `mosi_test/mosi_glove_results.csv` | MOSI transfer, GloVe fusions + GMTM |
| `mosi_test/ablation_mosi_results.csv` | MOSI transfer, BERT GMTM subsets |
| `mosi_test/ablation_mosi_glove_results.csv` | MOSI transfer, GloVe GMTM subsets |

`model/results/plot.ipynb` is an exploratory plotting notebook (large embedded
outputs). Prefer `examples/plot_published_results.py` for a headless redraw.
