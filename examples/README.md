# Examples

Small CPU programs that import the real modules in `model/models.py`
and run them on **synthetic** MOSI/MOSEI-shaped tensors. Nothing in
this directory downloads data or needs a GPU.

Run every command from the **repository root** so `examples.*` and
`model/models.py` both import cleanly:

```bash
python -m pip install -r requirements.txt
python examples/run_fusion_demo.py
python examples/run_gmtm_demo.py
python examples/run_metrics_demo.py
python examples/run_tiny_train.py
python -m unittest examples.test_metrics
```

`python -m examples.run_fusion_demo` works too. Each script puts the
repo root and `model/` on `sys.path` via `examples/repo.py`.

## What each file is for

| File | What it proves |
| --- | --- |
| `shapes.py` | The 35 / 74 / 768 (BERT) and 35 / 74 / 300 (GloVe) widths in one place |
| `synthetic_data.py` | How to draw a `[B, T, F]` clip and how ablation zeros a stream |
| `metrics.py` | Acc-7 / Acc-5 / Acc-2 / F1 / MAE / Pearson without a DataLoader |
| `run_fusion_demo.py` | Every fusion operator returns a finite tensor of the documented shape |
| `run_gmtm_demo.py` | GMTM forward is `[B, 1]`; unimodal zeros still run |
| `run_metrics_demo.py` | Perfect / shuffled / constant predictors behave as the protocol says |
| `run_tiny_train.py` | Gradients flow through a shrunk ConcatLate stack and GMTM |
| `compare_csv.py` | Diff a fresh sweep against `model/results/*.csv` |
| `repo.py` | `sys.path` helper so the scripts work as files or as modules |

## Flags worth knowing

```bash
# GloVe widths (text=300) instead of BERT (text=768)
python examples/run_fusion_demo.py --embedding glove
python examples/run_gmtm_demo.py --embedding glove

# Experimental GMTM size (4 layers, embed_dim=64)
python examples/run_gmtm_demo.py --full

# More SGD steps on the tiny train
python examples/run_tiny_train.py --steps 8 --seq-len 8

# Compare a new sweep to the snapshot
python examples/compare_csv.py \
    --snapshot model/results/main_results.csv \
    --fresh model/main_results.csv \
    --mae-tol 0.03
```

`--device auto` picks CUDA when it is visible; the default is CPU.

## What these examples deliberately are not

- They do **not** load `mosei_raw_bert.pkl` or any `.pt` checkpoint.
- They do **not** call `train()` / `test()` in `train_and_test.py`
  (those pull in `memory_profiler` and `plt.show()`).
- `run_tiny_train.py` uses smaller LSTM hidden sizes and a 1-layer
  GMTM. A loss that goes down on Gaussian noise is not evidence the
  MOSEI numbers will reproduce.
- Ablation in `run_gmtm_demo.py` zeros a stream on a **shared**
  random init. The CSV ablations are separately trained models.

## Adding another demo

1. Import `ensure_import_path` from `examples.repo` and call it
   before `import models`.
2. Draw tensors with `make_batch` / `make_vector_batch` so the shapes
   stay honest.
3. Fail loudly on non-finite values or unexpected ranks. Silent
   `NaN` is how a broken fusion change slips through.
4. Keep the demo runnable in a few seconds on CPU.
