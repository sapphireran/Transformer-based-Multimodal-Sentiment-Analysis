# Examples

These walkthroughs are the part of this personal repo that you can run
**without** CMU-MOSI / CMU-MOSEI downloads or a GPU. They import the real
classes from [`model/models.py`](../model/models.py) and feed them synthetic
tensors with the same ranks the pickle loaders emit after `max_pad=True`.

```bash
# from the repository root
python -m pip install -r requirements.txt
python -m examples.run_all
python -m pytest tests/ -q
```

Every module exposes `main() -> int` and is safe to invoke with
`python -m examples.<name>`.

## What each module is for

| Module | Reads real CSVs? | Uses GMTM? | Notes |
| --- | --- | --- | --- |
| [`synthetic_data`](synthetic_data.py) | no | no | `B×T×{35,74,768}` batches + pickle-shaped dicts |
| [`inspect_pickle_schema`](inspect_pickle_schema.py) | optional path | no | Prints the schema; validates a `.pkl` if you pass one |
| [`eval_metrics`](eval_metrics.py) | no | no | Headless Acc-2 / Acc-5 / Acc-7 / F1 / Pearson |
| [`metrics_walkthrough`](metrics_walkthrough.py) | no | no | Five gold/pred pairs, including a dropped-zero clip |
| [`results_tables`](results_tables.py) | yes | no | Lowest-MAE row in each committed experiment CSV |
| [`positional_embeddings`](positional_embeddings.py) | no | no | Detached sinusoidal table used inside GMTM |
| [`attention_pooling`](attention_pooling.py) | no | no | Softmax weights sum to 1 over time |
| [`encoder_stack`](encoder_stack.py) | no | no | LSTM / packed LSTM / GRU / TransformerSeq / MLP |
| [`fusion_shapes`](fusion_shapes.py) | no | no | Concat, TFN, LRTF, early/late transformer ranks |
| [`forward_gmtm`](forward_gmtm.py) | no | yes | Tiny GMTM (`embed=16`, 1 layer) → `[B, 1]` |
| [`ablation_zeroing`](ablation_zeroing.py) | no | yes | Same 7 subsets as `train_GMTM_glove.py` |
| [`tiny_train_loop`](tiny_train_loop.py) | no | yes | 2 epochs, AdamW, L1, grad-clip 8 |
| [`run_all`](run_all.py) | yes | yes | Sequential runner; non-zero if any step fails |

## Design choices

- **CPU only.** `examples.common.cpu_device()` never calls `.cuda()`.
- **Tiny HParams.** Training scripts use `embed_dim=64`, 4 layers, 4 heads.
  Examples use `TinyGMTMParams` (`16 / 1 / 2`) so a forward + two toy epochs
  finish in seconds. See [`docs/hyperparameters.md`](../docs/hyperparameters.md).
- **No `train_and_test.train()`.** That helper imports `memory_profiler` and
  opens a matplotlib window inside `single_test`. The toy trainer is a
  short explicit loop so the control flow is readable.
- **Ablations zero tensors** instead of shrinking the graph, matching
  `get_ablation_dataloader`.
- **Labels for the toy trainer** are `3 * tanh(mean(text))`, i.e. a text-driven
  target in `(-3, 3)`, not random noise.

## Passing a real pickle

If you have `model/data/MOSEI/mosei_raw_bert.pkl` locally:

```bash
python -m examples.inspect_pickle_schema model/data/MOSEI/mosei_raw_bert.pkl
```

The file is gitignored on purpose (hundreds of MB). The schema check only
looks at ranks and split keys.

## Reading the recorded experiments

```bash
python -m examples.results_tables
```

That printer is pinned to the headline MOSEI BERT GMTM MAE **0.5640** and to
`TransformerLate` as the best BERT baseline. If someone edits a CSV without
updating [`docs/experiments.md`](../docs/experiments.md), pytest fails.
