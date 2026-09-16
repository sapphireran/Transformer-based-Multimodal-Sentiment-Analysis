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
| `plot_results.py` | BERT vs GloVe line/bar charts from those CSVs (`examples/output/*.png`) |
| `bin_edges.py` | uniform Acc-7 / Acc-5 edges and boundary probes |
| `evaluate_toy.py` | MAE / Acc-7 / Acc-2 / F1 on synthetic labels (no torch models) |
| `metric_sensitivity.py` | bias / noise / `exclude_zero` effects on the same helpers |
| `packed_vs_padded.py` | `_process_1` vs `_process_2` collate on variable-length clips |
| `fusion_zoo.py` | forward pass of Concat / TFN / LMF / early+late Transformer |
| `count_params.py` | parameter counts at BERT and GloVe widths |
| `gmtm_forward.py` | one GMTM forward, parameter count, per-clip predictions |
| `train_toy_gmtm.py` | short AdamW + L1 loop on a tiny GMTM |
| `compare_toy_fusions.py` | mean-pool MLP vs tiny GMTM on one toy split |
| `ablate_toy_modalities.py` | zero-out the same 7 modality subsets the GMTM scripts use |
| `run_all.py` | runs the smoke list (not every script above) |

Shared helpers:

* `toy_data.py` — BERT (768-d) or GloVe (300-d) batches, optional masking
* `gmtm_forward.TinyHParams` — `embed_dim=16`, 1 layer, 2 heads
* `_common.py` — puts `model/` on `sys.path`

JSON dumps and PNGs land in `examples/output/` when you pass `--write-json`
(or when `plot_results.py` runs). That directory is gitignored.

## Typical invocations

```bash
python examples/inspect_results.py --only MOSI
python examples/plot_results.py
python examples/bin_edges.py
python examples/evaluate_toy.py --backend glove --batch-size 128
python examples/metric_sensitivity.py
python examples/packed_vs_padded.py
python examples/fusion_zoo.py --backend bert --seq-len 12
python examples/count_params.py --backend glove
python examples/gmtm_forward.py --preset tiny --backend glove
python examples/train_toy_gmtm.py --epochs 8 --backend glove
python examples/compare_toy_fusions.py --epochs 6
python examples/ablate_toy_modalities.py --batch-size 32
```

`--preset paper` in `gmtm_forward.py` builds the 64-d / 4-layer / 4-head
graph from `train_GMTM_bert.py`. It is much slower on CPU.

## Relation to the training scripts

| Training script | Example stand-in |
| --- | --- |
| `model/train_main_*.py` | `fusion_zoo.py`, `count_params.py` |
| `model/train_GMTM_*.py` | `gmtm_forward.py`, `train_toy_gmtm.py` |
| `get_ablation_dataloader` | `ablate_toy_modalities.py`, `ToyBatch.masked` |
| `train_and_test.single_test` | `evaluate_toy.py`, `metric_sensitivity.py` |
| `_process_1` / `_process_2` | `packed_vs_padded.py` |
| `model/results/plot.ipynb` | `plot_results.py` |
