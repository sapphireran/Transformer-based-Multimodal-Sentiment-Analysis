# Reproduction

Two tracks:

1. **Examples lab (this PR):** CPU, no pickles, no checkpoints. This is
   the path that is actually runnable in a fresh clone.
2. **Full MOSI / MOSEI training:** needs the CMU features, a GPU, and
   some script clean-up. Documented so a future personal run does not
   have to reverse-engineer paths from the original upload.

## Track 1 — examples lab

```bash
python -m pip install -r requirements.txt
python scripts/run_examples.py
python -m pytest
```

`scripts/run_examples.py` executes the six numbered scripts under
`examples/` in order and fails if the toy GMTM run does not reduce MAE.
None of this writes into `model/results/`.

## Track 2 — dataset pickles

1. Install [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).
2. Download MOSI / MOSEI high-level CSDs listed in `model/data/readme.md`.
3. Align to words (see `model/data/MOSEI/get_mosei.py` and the notebooks).
4. Export dicts of numpy arrays to:

   ```text
   model/data/MOSEI/mosei_raw_bert.pkl
   model/data/MOSEI/mosei_raw_glove.pkl
   model/data/MOSI/mosi_raw_bert.pkl
   model/data/MOSI/mosi_raw_glove.pkl
   ```

5. Work from `model/` so the `data.get_dataloader` imports resolve:

   ```bash
   cd model
   python train_main_bert.py
   python train_GMTM_bert.py
   ```

GloVe needs the real `glove.840B.300d.txt`; the file in git is empty.

## Hyper-parameters that match the CSVs (BERT MOSEI)

Taken from `train_main_bert.py` / `train_GMTM_bert.py`, not from the
function defaults in `train()` (those still say RMSprop / 0.001).

| item | value |
| --- | --- |
| batch size | 32 |
| max length | 50 |
| optimiser | AdamW |
| learning rate | 1e-4 |
| weight decay | 0.01 |
| loss | L1 |
| early stop | 7 epochs without valid improvement |
| GMTM embed / heads / layers | 64 / 4 / 4 |

Set `total_epochs` back to something like 20–50 before a serious run.
The committed `train_main_bert.py` uses `1` epoch.

## Checkpoint layout

```text
model/checkpoints/{FusionMethod}.pt          # BERT MOSEI baselines
model/checkpoints/glove_{FusionMethod}.pt    # GloVe MOSEI baselines
model/checkpoints/ablation/model_{a+b+c}.pt  # BERT GMTM ablations
model/checkpoints/ablation/model_glove_*.pt  # GloVe GMTM ablations
```

`torch.save(model, path)` stores the whole module. Loading needs the
same `models.py` class definitions and, in older PyTorch, the same
pickle protocol. `torch.load(..., weights_only=False)` is required on
PyTorch 2.6+.

MOSI eval scripts look at `../checkpoints/...` relative to
`model/mosi_test/`.

## GPU assumptions

Training scripts call `.cuda()` at construction time. There is no
`--device` flag. CPU-only machines should use the examples lab rather
than editing every constructor.

`LowRankTensorFusion` places its ones-vector on `cuda:0` when a GPU
exists. Keep fusion and data on the same device.

## Known script mismatches

See `docs/known-quirks.md`. The ones that block a naive re-run:

* GloVe `test(...)` extra kwargs (`dataset`, `no_robust`).
* `train_mosi_glove.py` imports `training_structures.Supervised_Learning`
  (MultiBench), which is not vendored here.
* GMTM scripts comment out `train()` and only test.

Fix those locally before trusting a new CSV against the tables in
`docs/experiments.md`.
