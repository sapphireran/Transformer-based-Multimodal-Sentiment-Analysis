# Examples

CPU walkthroughs that **do not** need `mosei_raw_*.pkl` or GPU checkpoints.
They import the real modules in `model/models.py` and the real CSVs in
`model/results/`.

Run them from either the repo root or this directory:

```bash
python -m pip install -r requirements-examples.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python examples/run_all.py
# or one at a time:
python examples/run_metrics.py
python examples/run_results_table.py
python examples/plot_published_results.py
python examples/run_fusion_forward.py
python examples/run_gmtm_toy.py
```

```bash
python -m pytest examples/tests -q
```

Outputs land in `examples/output/` (gitignored except you may want to keep
plots locally).

## What each script is for

| Script | Needs torch? | What it proves |
| --- | --- | --- |
| `run_metrics.py` | no | Uniform Acc-7 / Acc-5 edges and the binary F1 path match `docs/evaluation.md` |
| `run_results_table.py` | no | Every published CSV parses; BERT-MOSEI ranking facts still hold |
| `plot_published_results.py` | no | Headless bar charts of those CSVs |
| `run_fusion_forward.py` | yes | Concat / TFN / LMF / early & late transformer / GMTM accept the documented shapes and stay finite |
| `run_gmtm_toy.py` | yes | A tiny GMTM trains on synthetic labels and train L1 falls |
| `run_all.py` | yes | Runs the five scripts above in that order |

`run_gmtm_toy.py --keep text` zeros audio and vision, same protocol as
`get_ablation_dataloader`.

## Library modules

| Module | Role |
| --- | --- |
| `paths.py` | Repo root, `model/`, `examples/output/` |
| `device.py` | `cuda` if present, else `cpu` |
| `synthetic_affect.py` | In-memory pickle-shaped splits; labels ≈ `tanh(text)` |
| `metrics_lib.py` | `split_uniform_7/5`, `eval_affect`, `evaluate_affect_batch` |
| `results_lib.py` | CSV loader + name normalization (`Concat` → `ConcatLate`) |

`metrics_lib.py` is a deliberate copy of the formulas in
`train_and_test.py`, not an import of that file: importing
`train_and_test` requires `memory_profiler` and `single_test` calls
`plt.show()`.

## Config mirrors

`examples/configs/*.yaml` are **documentation**. The training scripts do
not read them. They exist so encoder widths, packed vs max-pad, and GMTM
`HParams` can be reviewed without opening four Python files.

| File | Mirrors |
| --- | --- |
| `configs/bert_mosei.yaml` | `train_main_bert.py` |
| `configs/glove_mosei.yaml` | `train_main_glove.py` |
| `configs/gmtm.yaml` | `train_GMTM_*.py` `HParams` |
| `configs/mosi_transfer.yaml` | `model/mosi_test/` |

## Tests

`examples/tests/` checks:

- bin edges and perfect / flipped metric cases
- synthetic array shapes and ablation zeros
- fusion output ranks (torch)
- CSV presence and the documented MAE inequalities
- YAML files still parse

They are unit tests for the **docs/examples contract**, not a re-run of
MOSEI.

## What these examples will not do

- They will not reach the MAE numbers in `model/results/`.
- They will not download CMU-MOSI / CMU-MOSEI.
- They will not load the gitignored `.pt` checkpoints.
- `run_gmtm_toy.py` uses `embed_dim=16`, `layers=1`. Published GMTM used
  `embed_dim=64`, `layers=4`.
