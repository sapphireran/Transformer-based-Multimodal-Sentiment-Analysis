# Personal examples

CPU-only demos that do **not** need MOSI/MOSEI pickles, GloVe, or a
GPU. They import the real modules under `model/` and run on tiny
synthetic tensors so the architecture and metrics can be inspected
without the CMU download.

```bash
# from the repository root
python3 -m pip install -r requirements.txt
python3 examples/run_all.py
```

| Script | What it shows |
| --- | --- |
| `synthetic_batch.py` | Factory for aligned `[B, T, F]` vision / audio / text + labels |
| `gmtm_forward.py` | Tiny GMTM forward pass and parameter count |
| `fusion_shapes.py` | Output ranks for concat / TFN / LRTF / transformer fusion |
| `metrics_walkthrough.py` | MAE, Pearson r, uniform Acc-7/5, exclude-zero Acc-2 / F1 |
| `summarize_logged_results.py` | Pretty-print every checked-in CSV and mark winners |
| `ablation_deltas.py` | MAE deltas: text-only vs +audio / +visual / full |
| `tiny_overfit.py` | A few Adam steps of GMTM on synthetic labels |
| `inspect_pickle.py` | If a local pickle exists, print split sizes and widths |
| `run_all.py` | Runs the demos that have no data dependency |

None of these write checkpoints or touch `model/checkpoints/`.
