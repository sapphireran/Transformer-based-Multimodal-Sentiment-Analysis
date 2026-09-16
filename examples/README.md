# Synthetic examples

None of these scripts need MOSI / MOSEI pickles or a GPU. They exist so
the personal architecture notes can be checked on a laptop.

Run them from the **repository root** so `examples/` is importable:

```bash
python3 -m pip install -r requirements.txt
python3 examples/fusion_forward.py
python3 examples/gmtm_forward.py
python3 examples/gmtm_toy_train.py
python3 examples/evaluate_metrics.py
python3 examples/packed_vs_padded.py
python3 examples/run_all.py
python3 -m pytest tests -q
```

| Script | What it proves |
| --- | --- |
| `synthetic_data.py` | Shared helpers: toy widths, two collates, zero-ablation |
| `fusion_forward.py` | Concat / tensor / transformer fusion output ranks |
| `gmtm_forward.py` | Pairwise cross-attention shapes + text-only ablation |
| `gmtm_toy_train.py` | 25 AdamW + L1 steps; loss should drop |
| `evaluate_metrics.py` | Acc7 / Acc5 / Acc2 on a 7-score sheet |
| `packed_vs_padded.py` | `_process_1` vs `_process_2` return signatures |
| `run_all.py` | Runs the five scripts above and exits non-zero on failure |

Toy feature widths are 8 / 10 / 16 with `T=12`, not 35 / 74 / 768. The
rank pattern is the same as the real padded loader. See
[docs/hyperparameters.md](../docs/hyperparameters.md) for the tiny
GMTM HParams.

`gmtm_toy_train.py` labels are a function of the **text** mean. A
collapse in train MAE means the graph is connected and gradients flow;
it is not a claim about MOSEI.
