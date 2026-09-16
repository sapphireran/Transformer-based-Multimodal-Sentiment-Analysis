# Reproduction vs. examples

Two different goals:

1. **Understand the code** — run `examples/`. No CMU data, no GPU, no
   checkpoints.
2. **Reproduce a CSV row** — you need the original pickles, a long train,
   and to repair the script mismatches listed in [quirks](quirks.md).

## Path A — examples (always available)

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

This checks:

- fusion modules accept MOSI-shaped batches and return finite tensors
- GMTM forward + zero-mask ablation runs on CPU
- Acc2 / Acc5 / Acc7 / MAE / Corr match the formulas in `eval_protocol.py`
- a toy L1 loop decreases loss on a synthetic text-correlated label

It does **not** reproduce `model/results/*.csv`. Those numbers are
dataset-scale.

## Path B — full MOSEI train (personal machine)

Checklist:

1. Download aligned pickles (or rebuild them with the SDK + notebooks).
2. Put them at `model/data/MOSEI/mosei_raw_bert.pkl` (and/or `_glove`).
3. `pip install -r requirements.txt` in an environment with CUDA if you
   want the original wall-clock.
4. `cd model`.
5. For GloVe / GMTM scripts, **uncomment** `train(...)`.
6. Remove `dataset=` / `no_robust=` kwargs from `test(...)` calls.
7. Raise `total_epochs` above 1 if you are on `train_main_bert.py`.
8. Make sure `plt.show()` has a display, or comment out the confusion
   matrix block.
9. Write checkpoints under `model/checkpoints/` (gitignores `*.pt`).

Approximate compute: GMTM (`embed_dim=64`, 4 layers, 9 cross-modal
encoders) on full MOSEI is the heavy job. The concat baselines are cheap.
TFN’s 128k-wide fused vector is memory-heavy at large batch sizes; the
scripts use batch 32.

## Path C — MOSI transfer

1. Finish Path B so `model/checkpoints/*.pt` exist.
2. Place `model/data/MOSI/mosi_raw_bert.pkl`.
3. Fix `train_mosi_glove.py` imports if you need the GloVe transfer script.
4. Remember the loader **merges** MOSI splits. Your number will match
   `mosi_*_results.csv` only if you keep that merge.

## What you should not expect

- Bit-identical MAE to the CSVs after a retrain (no logged seed, no
  `torch.use_deterministic_algorithms`, different PyTorch / CUDA).
- Loading an old `torch.save(model)` pickle on a much newer torch.
- Official MOSI leaderboard comparability from `get_mosi_dataloader`.

## Recording a new personal run

If you do retrain, write a new CSV next to the old one rather than
overwriting `model/results/*.csv`. Suggested columns (already used):

```
Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1
```

Add a one-line note in this file: date, pickle name, epochs, seed if any.
