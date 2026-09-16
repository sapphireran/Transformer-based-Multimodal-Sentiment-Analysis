# Examples

CPU-only demos that import the real modules under `model/` but **never** touch CMU pickles, GloVe, BERT weights, or `.pt` checkpoints. They exist so the personal architecture / metric notes can be executed on a laptop.

From the repo root:

```bash
python -m pip install -r requirements.txt
python examples/run_all_examples.py
```

Or one at a time:

| Command | What it shows |
| --- | --- |
| `python examples/fusion_forward.py` | Concat / TFN / LRTF / early+late transformer on a synthetic BERT batch |
| `python examples/gmtm_forward.py` | GMTM BERT + GloVe, including a zeroed-modality ablation |
| `python examples/metrics_demo.py` | `eval_affect` / `split_uniform_*` on five hand-built scores |
| `python examples/summarize_results.py` | Markdown reprint of the eight committed CSVs |

`summarize_results.py` writes `examples/output/result_tables.md` (gitignored). The other demos only print.

## Shapes the fusions see

`common.make_synthetic_batch` builds the same last-axis widths as the loaders:

```text
vision [B, 50, 35]
audio  [B, 50, 74]
text   [B, 50, 768]   # or 300 for GloVe
label  [B, 1]         # clipped to [-3, 3]
```

`zero_modalities(batch, keep=("text",))` clones that batch and writes zeros into the dropped streams — the same trick `get_ablation_dataloader` uses, so a “text-only” GMTM still has three towers.

## What is deliberately not here

- A training loop. `train()` wants real dataloaders and (as committed) a GPU.
- Checkpoint loading. There are no `.pt` files in git.
- MOSI/MOSEI downloads. See [docs/reproducing.md](../docs/reproducing.md) for that path.

Pytest wraps the same functions:

```bash
python -m pytest tests/test_examples.py -q
```
