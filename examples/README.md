# Examples

These scripts do **not** need the CMU-MOSI / CMU-MOSEI pickles. They rebuild
the tensor shapes the training code expects and run on CPU.

```bash
pip install -r requirements-dev.txt
python examples/run_all.py
```

| Script | What it shows |
| --- | --- |
| `inspect_results.py` | reprints the committed CSVs and stars the best cell in each column |
| `evaluate_toy.py` | MAE / Acc-7 / Acc-2 / F1 on synthetic labels (no torch models) |
| `fusion_zoo.py` | forward pass of Concat / TFN / LMF / early+late Transformer |
| `gmtm_forward.py` | one GMTM forward, parameter count, per-clip predictions |
| `train_toy_gmtm.py` | short AdamW + L1 loop on a tiny GMTM |
| `ablate_toy_modalities.py` | zero-out the same 7 modality subsets the GMTM scripts use |
| `run_all.py` | runs the list above |

Shared helpers:

* `toy_data.py` — BERT (768-d) or GloVe (300-d) batches, optional masking
* `gmtm_forward.TinyHParams` — `embed_dim=16`, 1 layer, 2 heads
* `_common.py` — puts `model/` on `sys.path`

JSON dumps land in `examples/output/` when you pass `--write-json`.

## Typical invocations

```bash
python examples/inspect_results.py --only MOSI
python examples/evaluate_toy.py --backend glove --batch-size 128
python examples/fusion_zoo.py --backend bert --seq-len 12
python examples/gmtm_forward.py --preset tiny --backend glove
python examples/train_toy_gmtm.py --epochs 8 --backend glove
python examples/ablate_toy_modalities.py --batch-size 32
```

`--preset paper` in `gmtm_forward.py` builds the 64-d / 4-layer / 4-head
graph from `train_GMTM_bert.py`. It is much slower on CPU.

## Relation to the training scripts

| Training script | Example stand-in |
| --- | --- |
| `model/train_main_*.py` | `fusion_zoo.py` |
| `model/train_GMTM_*.py` | `gmtm_forward.py`, `train_toy_gmtm.py` |
| `get_ablation_dataloader` | `ablate_toy_modalities.py`, `ToyBatch.masked` |
| `train_and_test.single_test` | `evaluate_toy.py` + `model/metrics.py` |
