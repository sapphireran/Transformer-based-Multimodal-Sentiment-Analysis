# Repo map

Personal index of what each path is for. Paths are from the repository
root.

## Root

| Path | What it is |
| --- | --- |
| `README.md` | Short project guide + headline tables |
| `LICENSE` | MIT, copyright 2024 `pang990801` |
| `requirements.txt` | Python deps for training and examples |
| `docs/` | These notes |
| `examples/` | Synthetic, CPU-only walkthroughs |
| `model/` | Original experiment code and CSVs |

## `model/` — training entry points

| Path | What it does |
| --- | --- |
| `models.py` | All `nn.Module`s (encoders, fusion, GMTM) |
| `train_and_test.py` | `MultiFramework`, `train`, `test`, metrics |
| `train_main_bert.py` | Six fusion methods on MOSEI BERT |
| `train_main_glove.py` | Same sweep, GloVe (train call commented out) |
| `train_GMTM_bert.py` | GMTM + modality list, BERT |
| `train_GMTM_glove.py` | GMTM + full 7-way ablation, GloVe |
| `main_results.csv` / `glove_results.csv` | Copies of the main sweeps |
| `ablation_results.csv` / `ablation_glove_results.csv` | GMTM ablations |

## `model/results/`

Tidier copies of the CSVs plus `plot.ipynb` (large, embedded figures from
the original plotting session). Prefer the CSVs; the notebook is an
artifact, not a source of truth. Index: [`model/results/README.md`](../model/results/README.md).

## `model/data/`

| Path | What it does |
| --- | --- |
| `get_dataloader.py` | Pickle → `DataLoader`, ablation zeros, collate |
| `readme.md` | Original one-screen file checklist |
| `analysis_dataset.ipynb` | Peeked at pickle keys / shapes |
| `MOSEI/get_mosei.py` | `mmsdk` alignment sketch (Windows paths) |
| `MOSEI/get_dataset.ipynb` | Interactive alignment |
| `MOSI/get_dataset.ipynb` | Same for MOSI |

Pickles themselves are not in git.

## `model/mosi_test/`

Transfer evaluation: load a **MOSEI** checkpoint, score a **merged MOSI**
pool.

| Path | Notes |
| --- | --- |
| `train_mosi_bert.py` | Fusion family, BERT features |
| `train_mosi_glove.py` | Imports `training_structures.Supervised_Learning` — MultiBench leftover, will fail in this repo unless you change the import |
| `mult_bert_mosi.py` | GMTM BERT transfer |
| `mult_glove_mosi.py` | GMTM GloVe transfer |
| `*_results.csv` | Numbers copied into `docs/experiments.md` |

## `model/checkpoints/`

Only readmes. Expected names if you train locally:

```
model/checkpoints/
  ConcatEarly.pt
  ConcatLate.pt
  LowRankTensorFusion.pt
  TensorFusion.pt
  TransformerEarly.pt
  TransformerLate.pt
  glove_<Fusion>.pt
  ablation/
    model_<mod>+<mod>.pt
    model_glove_<mod>+<mod>.pt
```

GMTM names use `'+'.join(modalities)`, e.g. `model_text+audio+visual.pt`.

## `examples/`

| Path | What it does |
| --- | --- |
| `common.py` | `sys.path` + shared constants |
| `synthetic_multimodal.py` | Tiny MOSI-shaped tensors / optional pickle |
| `eval_protocol.py` | MAE / Corr / Acc2 / Acc5 / Acc7 / F1 |
| `demo_fusion.py` | Forward each fusion module |
| `demo_gmtm.py` | GMTM forward + tiny ablation |
| `demo_metrics.py` | Walk the binning rules on toy scores |
| `demo_train_toy.py` | Few-step L1 train on synthetic data |
| `inspect_shapes.py` | Print shapes through the BERT-sized stack |
| `print_logged_results.py` | Pretty-print every checked-in results CSV |
| `run_all.py` | Run every demo and the unit checks |
| `test_examples.py` | Assertions used by `run_all` |

## What is *not* here

- No pretrained `.pt` weights
- No MOSI/MOSEI pickles or `.csd` files
- No GloVe `840B.300d` vectors
- No MultiBench / `training_structures` package (one MOSI script still
  tries to import it)
