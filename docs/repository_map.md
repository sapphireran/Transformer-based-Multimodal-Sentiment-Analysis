# Repository map

Personal layout after the docs / examples expansion. Paths are relative to
the repo root.

## Top level

| Path | Notes |
| --- | --- |
| `README.md` | Project overview, tables, quick start |
| `LICENSE` | MIT, copyright pang990801 |
| `requirements.txt` | torch, sklearn, pytest, … |
| `.gitignore` | pickles, `.csd`, weights, caches |
| `docs/` | Long-form notes |
| `examples/` | Synthetic walkthroughs |
| `tests/` | pytest for examples + CSV helpers |
| `model/` | Original experiment code (unchanged logic) |

## `docs/`

| File | Topic |
| --- | --- |
| `architecture.md` | GMTM + six baselines, tensor ranks |
| `datasets.md` | MOSI / MOSEI, pickle schema, ablation zeros |
| `metrics.md` | MAE, Corr, Acc-2/5/7, F1 |
| `experiments.md` | All committed CSV tables + reading notes |
| `training.md` | `MultiFramework` loop, optimizer, scripts |
| `hyperparameters.md` | HParams and encoder widths |
| `reproduction.md` | How to re-run, known script mismatches |
| `module_reference.md` | Class index |
| `glossary.md` | Short definitions |

## `examples/`

Importable as `python -m examples.<name>` from the repo root.

| Module | What it demonstrates |
| --- | --- |
| `common` | `sys.path`, CPU device, tiny vs train HParams |
| `synthetic_data` | Batches and pickle-shaped dicts |
| `eval_metrics` | Headless copy of the eval formulas |
| `forward_gmtm` | GMTM parameter count + `[B,1]` output |
| `fusion_shapes` | Concat / TFN / LRTF / transformers |
| `attention_pooling` | Softmax weights over time |
| `positional_embeddings` | Sinusoidal table + `make_positions` |
| `ablation_zeroing` | Same zero-fill as `get_ablation_dataloader` |
| `tiny_train_loop` | Two-epoch AdamW + L1 on synthetic data |
| `results_tables` | Parse committed CSVs, pick winners |
| `encoder_stack` | LSTM / GRU / TransformerSeq ranks |
| `inspect_pickle_schema` | Document / optionally validate a `.pkl` |
| `run_all` | Sequential runner used by CI-style checks |

## `model/` (original tree)

```
model/
├── models.py
├── train_and_test.py
├── train_main_bert.py
├── train_main_glove.py
├── train_GMTM_bert.py
├── train_GMTM_glove.py
├── main_results.csv              # copies also live under results/
├── glove_results.csv
├── ablation_results.csv
├── ablation_glove_results.csv
├── data/
│   ├── get_dataloader.py
│   ├── readme.md
│   ├── analysis_dataset.ipynb
│   ├── MOSEI/get_mosei.py
│   ├── MOSEI/get_dataset.ipynb
│   └── MOSI/get_dataset.ipynb
├── mosi_test/
│   ├── train_mosi_bert.py
│   ├── train_mosi_glove.py
│   ├── mult_bert_mosi.py
│   ├── mult_glove_mosi.py
│   └── *results.csv
├── results/
│   ├── main_results.csv
│   ├── glove_results.csv
│   ├── ablation_results.csv
│   ├── ablation_glove_results.csv
│   └── plot.ipynb
└── checkpoints/
    ├── readme.md
    └── ablation/readme.md
```

## What was intentionally left alone

- No company code was imported.
- Training scripts were not rewritten to “fix” commented `train()` calls
  or extra `test()` kwargs; those mismatches are listed in
  [`reproduction.md`](reproduction.md) instead.
- Social/media integrations are not used.
