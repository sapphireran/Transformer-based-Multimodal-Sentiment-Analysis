# Examples

CPU walkthroughs that do **not** need official MOSI / MOSEI pickles or a GPU.
They use the same class names and metric helpers as the real trainers.

```bash
# from the repository root
python -m pip install -r requirements-examples.txt
python examples/01_synthetic_mosi_dataset.py
python examples/02_fusion_forward_pass.py
python examples/03_gmtm_tiny_train.py
python examples/04_metrics_walkthrough.py
python examples/05_results_tables.py
```

| Script | What it proves |
| --- | --- |
| `01_synthetic_mosi_dataset.py` | MOSI pickle schema + `get_dataloader(max_pad=True)` |
| `02_fusion_forward_pass.py` | Every fusion class accepts a toy batch |
| `03_gmtm_tiny_train.py` | Identity → GMTM → Identity, L1, Acc2/F1 |
| `04_metrics_walkthrough.py` | Uniform Acc7 vs rounded Acc7, exclude-zero Acc2 |
| `05_results_tables.py` | Reprint the eight checked-in result CSVs |

`--full-width` on script 01 builds a `(50, 35 / 74 / 768)` pickle, the same
shapes as `mosi_raw_bert.pkl`. The default is a tiny `(8, 8 / 10 / 16)` bundle
so GMTM steps stay cheap.

Generated pickles land in `examples/_generated/` and are gitignored.

## What these are not

- Not published MOSI scores. Labels are a function of the synthetic text mean.
- Not a replacement for `model/train_GMTM_bert.py` or `model/mosi_test/`.
- Not a download of CMU-MOSI. See [`docs/datasets.md`](../docs/datasets.md).

The next *real* MOSI experiment (train on the MOSI train fold, score the test
fold) is written up in [`docs/mosi.md`](../docs/mosi.md).
