# Examples

CPU walkthroughs for the personal multimodal sentiment code. **No MOSI /
MOSEI download** and no GPU. They import building blocks from
`model/models.py` and reimplement the metric helpers so nothing opens a
matplotlib window.

Run everything from the repository root:

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

Or run one file:

```bash
python examples/synthetic_data.py
python examples/packed_vs_padded.py
python examples/encoder_shapes.py
python examples/fusion_walkthrough.py
python examples/metrics_demo.py
python examples/result_tables.py
python examples/gmtm_toy_train.py
```

`examples/test_examples.py` is a unittest wrapper around the same checks.

## What each script shows

| Script | What you should see |
| --- | --- |
| `synthetic_data.py` | BERT- and GloVe-shaped utterances, labels on `[-3, 3]`, optional zeroed streams |
| `packed_vs_padded.py` | `_process_1`-style packed batch vs `_process_2`-style `T=50` batch |
| `encoder_shapes.py` | LSTM / GRU / `TransformerSeq` input and output ranks |
| `fusion_walkthrough.py` | Concat, TFN, LMF, early/late transformer output shapes |
| `metrics_demo.py` | MAE, Corr, Acc7/5/2, F1 on a tiny hand-labeled set |
| `result_tables.py` | rank the checked-in CSVs (star = best cell) |
| `gmtm_toy_train.py` | GMTM overfits synthetic labels for a few Adam steps |

## Design constraints

- Device is `cuda` only if `torch.cuda.is_available()`, otherwise CPU.
- Feature widths match the training scripts: visual 35, audio 74, text 768
  or 300.
- GMTM is constructed with `n_modalities=3` even in “text-only” toys, with
  zeros in the other slots — same protocol as `get_ablation_dataloader`.
- These scripts do not write checkpoints and do not load `.pkl` files.

## After you have real data

The toys stop at shapes and a sanity backward pass. Real training is
documented in [`../docs/training.md`](../docs/training.md) and started from
`model/` with the pickle paths in [`../docs/datasets.md`](../docs/datasets.md).
