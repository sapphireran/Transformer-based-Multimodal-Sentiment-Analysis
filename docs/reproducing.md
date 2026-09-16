# Reproducing the original GPU runs

The `examples/` tree is enough to understand modules and metrics. This page is only for rebuilding MOSI/MOSEI pickles and rerunning `model/train_*.py` the way the CSVs were produced.

## 1. Environment

```bash
python -m pip install -r requirements.txt
# plus a CUDA wheel of torch that matches your GPU
# plus CMU-MultimodalSDK: https://github.com/A2Zadeh/CMU-MultimodalSDK
```

Python 3.10–3.12 is fine for the synthetic examples. The original notebooks were run on Windows with local `F:\MOSEI\...` paths.

## 2. Download computational sequences

From the SDK / CMU Multimodal Data SDK mirrors, place the `.csd` files listed in [datasets.md](datasets.md) under:

```
model/data/MOSEI/cmumosei/
model/data/MOSI/cmumosi/
```

Download `glove.840B.300d.txt` if you want GloVe pickles. Replace the empty placeholder at `model/data/glove.840B.300d.txt`.

## 3. Align and pickle

1. Edit paths in `model/data/MOSEI/get_mosei.py` and the `get_dataset.ipynb` notebooks (they still point at the original machine).
2. Align words → visual / acoustic (mean pool) → labels.
3. Write `mosei_raw_bert.pkl`, `mosei_raw_glove.pkl`, `mosi_raw_bert.pkl`, `mosi_raw_glove.pkl` next to those folders.
4. Sanity-check with `model/data/analysis_dataset.ipynb` (expect keys `train` / `valid` / `test` and text width 768 or 300).

## 4. Train

```bash
cd model
# BERT bake-off (train() is live; epochs=1 in the file — raise it)
python train_main_bert.py

# BERT GMTM: uncomment train(...) first
python train_GMTM_bert.py
```

Uncomment the corresponding `train(` blocks in the GloVe scripts before a from-scratch GloVe run. Checkpoint directories:

```
model/checkpoints/
model/checkpoints/ablation/
```

## 5. MOSI transfer

Keep the MOSEI `.pt` files where the scripts look (`../checkpoints/...` from `model/mosi_test/`). Then:

```bash
cd model/mosi_test
python train_mosi_bert.py
python mult_glove_mosi.py
```

`train_mosi_glove.py` still imports `training_structures.Supervised_Learning` (a MultiBench leftover). Point it at `train_and_test` like the BERT file before running.

## 6. Compare to the committed CSVs

`python examples/summarize_recorded_results.py` prints the committed tables. New runs will not match bit-for-bit (AdamW, dropout, GPU reductions). Use MAE / Acc-2 / Corr as the comparison, not exact floats.

## What you cannot reproduce from this git clone alone

- The `.pt` weights (never committed)
- The raw `.csd` / `.pkl` / GloVe vectors
- The exact CUDA / PyTorch build used in 2024

You **can** reproduce module shapes, metric definitions, and a toy optimization path with `examples/` and `tests/`.
