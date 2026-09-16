# Personal examples (no CMU downloads)

These scripts import the real modules in `model/models.py` and
`model/train_and_test.py`. They **do not** read `.pkl` / `.csd` files.
Every tensor is allocated in `common.make_aligned_batch` with the MOSI /
MOSEI feature widths (35 / 74 / 768 or 300).

```bash
python -m pip install -r requirements.txt
python examples/01_synthetic_batch.py
python examples/02_fusion_forward.py
python examples/03_gmtm_forward.py
python examples/04_metrics_walkthrough.py
python examples/05_mini_training.py
python examples/06_result_tables.py
```

Run them from **either** the repo root or `examples/`. `common.py` puts
`model/` on `sys.path`.

| Script | What it shows |
| --- | --- |
| `01_synthetic_batch.py` | BERT vs GloVe shapes; correlated labels |
| `02_fusion_forward.py` | Concat / TF / LRTF / both transformers, one forward each |
| `03_gmtm_forward.py` | Tiny GMTM (2 layers, embed 16) + param count |
| `04_metrics_walkthrough.py` | Acc-7 edges, zero-exclusion, toy MAE / Corr |
| `05_mini_training.py` | 8-epoch ConcatLate + AdamW + L1 on synthetic text |
| `06_result_tables.py` | Rank every checked-in CSV by MAE |

`metrics.py` is the headless subset of `single_test` (no `plt.show()`).

## Design choices

- **CPU only.** `common.device_of()` never picks CUDA, so the examples
  match this workspace.
- **Shrunken width where it is only a demo.** ConcatLate’s text LSTM is
  128-d here, not 1024-d. GMTM uses `TinyHParams` instead of the training
  `HParams`. Comments mark the production numbers.
- **Honest mismatches.** `02_fusion_forward.py` prints the BERT
  `MLP(64, …)` vs 32-d `EarlyFusionTransformer` size error instead of
  papering over it.
- **Planted text signal.** `correlated=True` adds a clip-level latent to
  every text frame. Mean-pooling raw Gaussian BERT noise is too weak for
  an 8-epoch demo.
- **Packed lengths** for LSTM/GRU demos are “every clip has length T”.
  That is enough to exercise `has_padding=True` without a real collate.

## Adding another example

1. `from common import make_aligned_batch` (and import models after that,
   so `sys.path` is already patched).
2. Keep side effects to stdout. Do not write checkpoints unless you put
   them under `examples/_outputs/` (gitignored).
3. If you need Acc-7 / F1, import `examples.metrics`, not
   `train_and_test.single_test` (that one opens a GUI confusion matrix).
