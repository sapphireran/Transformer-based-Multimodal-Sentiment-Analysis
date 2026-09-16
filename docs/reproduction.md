# Reproduction checklist

This is the shortest path from a clean clone to numbers that can be
compared with `model/results/*.csv`. Examples (no dataset) are step 0
and should pass on CPU in a couple of minutes.

## 0. Synthetic sanity (no MOSI / MOSEI)

```bash
python -m pip install -r requirements.txt
python examples/run_fusion_demo.py
python examples/run_gmtm_demo.py
python examples/run_metrics_demo.py
python examples/run_tiny_train.py
```

What "pass" means:

- fusion demo prints an output shape for each operator and does not
  raise
- GMTM demo prints a `[B, 1]` prediction and a finite L1
- metrics demo prints a seven-key report whose Acc-2 is in `[0, 1]`
- tiny-train demo's last loss is finite and usually lower than the
  first

If these fail, stop. The problem is the environment or a change in
`model/models.py`, not the dataset.

## 1. Python environment for real training

- Python 3.10+
- PyTorch 2.x with CUDA if you want the published-size models
- `pip install -r requirements.txt`
- optional: `mmsdk` (CMU Multimodal SDK) only if you rebuild pickles
  from `.csd`

Confirm:

```bash
python -c "import torch, sklearn, scipy; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Dataset files

You need **one** of these pairs, depending on which script you run:

| Script family | Pickle |
| --- | --- |
| `train_main_bert.py`, `train_GMTM_bert.py` | `model/data/MOSEI/mosei_raw_bert.pkl` |
| `train_main_glove.py`, `train_GMTM_glove.py` | `model/data/MOSEI/mosei_raw_glove.pkl` |
| `mosi_test/*bert*` | `model/data/MOSI/mosi_raw_bert.pkl` |
| `mosi_test/*glove*` | `model/data/MOSI/mosi_raw_glove.pkl` |

To rebuild from official computational sequences:

1. Download the four `.csd` files per corpus (see
   [datasets.md](datasets.md)).
2. Put them under `model/data/MOSEI/cmumosei/` or
   `model/data/MOSI/cmumosi/`.
3. For GloVe, download `glove.840B.300d.txt` into `model/data/`.
4. Run the matching `get_dataset.ipynb` after fixing the local paths.

Sanity-check a pickle before training:

```python
import pickle
from pathlib import Path
p = Path("model/data/MOSEI/mosei_raw_bert.pkl")
data = pickle.load(p.open("rb"))
for split in ("train", "valid", "test"):
    print(split, {k: data[split][k].shape for k in data[split]})
```

You want `vision[..., 35]`, `audio[..., 74]`, `text[..., 768]`, and
a label array whose first dimension matches.

## 3. Reproduce a single baseline (recommended first)

`train_main_bert.py` is the only sweep script that still calls
`train()`. Edit it so you are not launching all six methods at epoch
count 1:

```python
fusion_methods = ["ConcatLate"]
# in run_experiment:
train(..., 15, ...)   # was 1
```

Then:

```bash
cd model
python train_main_bert.py
```

You should see a `checkpoints/ConcatLate.pt` and a one-row
`main_results.csv`. MAE on MOSEI-BERT for ConcatLate was 0.6148 in
the snapshot; a 15-epoch retrain should land in a similar band, not
necessarily on top of it.

## 4. Reproduce GMTM

1. Uncomment the `train(...)` block in `train_GMTM_bert.py`.
2. Restore the 7-row `modality_combinations` list if you want the
   full ablation; leave only `['text', 'audio', 'visual']` for the
   headline run.
3. Create `model/checkpoints/ablation/` if it is missing.
4. `python train_GMTM_bert.py` from `model/`.

Expected headline (BERT, trimodal): MAE ≈ 0.564, Acc-2 ≈ 0.843,
Corr ≈ 0.726. Single-seed drift of ±0.01 MAE is normal.

## 5. MOSI transfer

Requires the MOSEI checkpoint **and** the MOSI pickle.

```bash
cd model/mosi_test
python train_mosi_bert.py      # fusion checkpoints
python mult_bert_mosi.py       # GMTM
```

`train_mosi_glove.py` currently imports
`training_structures.Supervised_Learning`. Point that import at
`train_and_test` (as `train_mosi_bert.py` already does) before you
rely on it.

Remember: `get_mosi_dataloader` scores train+valid+test MOSI, so
these numbers are not the official MOSI test split.

## 6. Compare against the snapshot

```bash
python examples/compare_csv.py \
    --snapshot model/results/main_results.csv \
    --fresh model/main_results.csv \
    --mae-tol 0.03
```

The helper flags any row whose MAE drifted more than `--mae-tol` or
whose method name is missing. It does not prove correctness; it
catches "the sweep silently skipped TensorFusion" class mistakes.

## 7. Known footguns

- **`.cuda()` at import / construct time.** CPU boxes die before the
  first batch. Use `.to(device)`.
- **`torch.save(model)` whole-module pickles.** Reloading needs the
  same class definitions on `sys.path`. A `state_dict` would be
  safer for long-term storage; the existing scripts do not do that.
- **`plt.show()` inside `single_test`.** Headless servers hang.
  `MPLBACKEND=Agg` or comment the block.
- **`memory_profiler`.** Required when `track_complexity=True`
  (the default).
- **GloVe `TransformerLate` `in_dim=1792`.** The live encoder widths
  sum to 704. A fresh (untrained) GloVe late-fusion model must use
  the real concatenated width; the 1792 value is for the saved
  checkpoint only.
- **Name collision `Linear` in `models.py`.** The function wins.
  Do not `from models import Linear` expecting the class.
- **MOSI glove test script** still talks to MultiBench's
  `training_structures` package.

## 8. What this repo does not reproduce

- Official leaderboard numbers from Tsai et al. / Zadeh et al. / later
  papers. The Acc-7 binning here is equal-width, not integer rounding.
- Multi-seed confidence intervals.
- End-to-end training from raw mp4. Features are precomputed.
- A packaged pip install. Import the `model/` directory or add it to
  `PYTHONPATH`.
