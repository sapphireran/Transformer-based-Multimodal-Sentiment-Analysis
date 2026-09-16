# Examples

Runnable CPU demos that **do not** download MOSI, MOSEI, GloVe, or
BERT. They import `model/models.py`, `model/metrics.py`, and
`model/synthetic.py` only.

```bash
# from the repository root
python -m pip install -r requirements-examples.txt
python examples/forward_fusion.py
python examples/train_toy_gmtm.py
python examples/metrics_demo.py
python examples/zero_modality_ablation.py
python examples/inspect_results.py
python -m pytest tests -q
```

| Script | What you should see |
| --- | --- |
| `forward_fusion.py` | One forward pass per fusion graph; printed output shapes |
| `train_toy_gmtm.py` | A few AdamW steps on synthetic clips; train MAE falling |
| `metrics_demo.py` | MAE / Acc-7 / Acc-2 / F1 on a known toy prediction |
| `zero_modality_ablation.py` | GMTM logits with text/audio/visual zeroed out |
| `inspect_results.py` | Ranked reprint of every checked-in CSV |

Output artifacts (if any) go to `examples/output/`, which is
gitignored.

These runs are smoke tests for the personal research code. They are
not MOSI/MOSEI scores — see [`docs/results.md`](../docs/results.md)
for those.
