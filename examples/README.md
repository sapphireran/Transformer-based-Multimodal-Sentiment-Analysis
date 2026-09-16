# Personal examples (no MOSI/MOSEI download)

These scripts exercise the modules in `model/models.py` on **synthetic**
clips that have the same feature widths as the real pickles (35 / 74 /
768 or a shrunken text dim). They are the path that always runs on a
laptop. They do not reproduce `model/results/*.csv`.

Background: [`docs/reproduction.md`](../docs/reproduction.md).

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

Each demo is also a standalone script. `sys.path` is adjusted so
`import models` resolves to `model/models.py`.

```bash
python examples/inspect_shapes.py
python examples/demo_fusion.py
python examples/demo_gmtm.py
python examples/demo_metrics.py
python examples/demo_train_toy.py
python examples/print_logged_results.py
python examples/test_examples.py
```

GPU is optional. `common.device()` uses `cuda:0` when it exists.

## What each file is for

| File | What you should see |
| --- | --- |
| `synthetic_multimodal.py` | Builds a pickle-compatible `{train,valid,test}` dict. Labels in `[-3, 3]`. Optional text-correlated signal for the toy trainer. |
| `eval_protocol.py` | MAE / Corr / Acc2 / Acc5 / Acc7 / F1, same rules as `single_test` without `plt.show()`. |
| `inspect_shapes.py` | Prints `[B, T, F]` through concat, TFN, LMF, both transformers, GMTM. |
| `demo_fusion.py` | One forward pass per main-sweep fusion family. Prints parameter counts (TFN vs LMF is the interesting contrast). |
| `demo_gmtm.py` | Seven zero-masks, one GMTM graph. Parameter count must stay constant. |
| `demo_metrics.py` | Bin edges, a few `(y, ŷ)` pairs, perfect / sign / zero / shuffled predictors. |
| `demo_train_toy.py` | 8 AdamW + L1 steps on synthetic text-correlated labels. Train L1 must drop; test MAE must beat a constant-mean baseline. Writes `examples/output/toy_train_metrics.txt`. |
| `print_logged_results.py` | Pretty-print every checked-in CSV and name the lowest-MAE row. |
| `test_examples.py` | Assertions for schema, binning, fusion smoke, GMTM masks, CSV tables. |
| `run_all.py` | All of the above in one process. |

## Design choices worth knowing

- **Zero-mask ablation, not a smaller net.** `demo_gmtm.py` uses the same
  trick as `get_ablation_dataloader`. A “text-only” batch still has visual
  and audio slots, filled with zeros.
- **Shrunken time / text for CPU.** Shape demo keeps 768-d text and `T=6`.
  GMTM / toy train use `text_dim=32` and `T≈12` so a CPU pass is seconds,
  not minutes. Swap those constants if you want a heavier smoke test.
- **Late transformer width is consistent here.** The demo concatenates
  `16+24+32` and builds `LateFusionTransformer(in_dim=72)`. That avoids
  the GloVe script mismatch documented in [`docs/quirks.md`](../docs/quirks.md).
- **Toy train is a graph check.** A 80-clip Gaussian toy set is not MOSI.
  If you change `TinyHParams` and the loss no longer drops, you broke
  the backward path — that is the only failure this demo is allowed to
  claim.

## Optional: write a fake pickle

```python
from examples.synthetic_multimodal import SyntheticConfig, save_synthetic_pickle

save_synthetic_pickle("/tmp/toy_mosei.pkl", SyntheticConfig(text_dim=768))
```

`get_dataloader` can read that path. Do not expect useful sentiment
metrics from it; the “text” channel is random noise plus a tanh of its
mean.
