# Examples

Runnable walkthroughs for this personal MOSI/MOSEI study repo. None of these scripts download data or call social APIs. They use the committed CSV tables and **synthetic** tensors that match the real feature widths.

```bash
# from the repo root
python examples/summarize_recorded_results.py
python examples/metrics_walkthrough.py
python examples/fusion_forward_pass.py
python examples/gmtm_forward_pass.py
python examples/gmtm_forward_pass.py --ablate text
python examples/toy_train_loop.py
python examples/model_inventory.py
python -m pytest tests/ -q
```

| Script | What you should see |
| --- | --- |
| `summarize_recorded_results.py` | All committed result tables, with best MAE / Acc-2 / Corr marked |
| `metrics_walkthrough.py` | Uniform 7-bin edges and every metric on a 12-point toy pair |
| `fusion_forward_pass.py` | Output shapes for concat, tensor, low-rank, and transformer fusions |
| `gmtm_forward_pass.py` | GMTM `[B, 1]` preds; `--ablate` zeros streams like the real loader |
| `toy_train_loop.py` | A few AdamW steps on a tiny GMTM; train L1 should fall |
| `model_inventory.py` | Parameter counts for GMTM (BERT vs GloVe widths) and bake-off heads |

Library code (imported by the scripts and by `tests/`):

| Module | Role |
| --- | --- |
| `examples/lib/paths.py` | Repo root and `sys.path` hook for `model/` |
| `examples/lib/synthetic.py` | MOSI/MOSEI-shaped random batches |
| `examples/lib/metrics.py` | MAE / Corr / Acc-2 / Acc-5 / Acc-7 / F1 (no matplotlib) |
| `examples/lib/results_tables.py` | CSV reader + “best cell” highlighter |

These examples are **not** a replacement for `model/train_GMTM_bert.py`. They exist so the architecture and metric definitions can be exercised in a clone that has no `.pkl` files.
