# Tests

```bash
python -m pip install -r requirements-examples.txt
python -m pytest tests -q
```

| File | Coverage |
| --- | --- |
| `test_metrics.py` | Acc7/5/2, exclude-zero F1, Pearson edge cases |
| `test_synthetic_dataset.py` | MOSI-shaped pickle + `get_dataloader` |
| `test_fusion_forward.py` | Concat / TFN / LMF / transformers / GMTM |
| `test_results_tables.py` | The eight checked-in experiment CSVs still parse |

No official dataset download and no GPU. `test_fusion_forward.py` and the
loader test skip or fail only if `torch` is missing.
