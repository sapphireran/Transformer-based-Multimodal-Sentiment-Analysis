# Script map

All paths are relative to the repo root. Run the training scripts with `cwd=model/` so the `data/...` and `checkpoints/...` relative paths resolve.

## MOSEI — fusion bake-off

| Script | Text | Does |
| --- | --- | --- |
| `model/train_main_bert.py` | BERT 768 | Train (1 epoch in the checked-in file) + test concat / tensor / transformer fusions. Writes `model/main_results.csv`. |
| `model/train_main_glove.py` | GloVe 300 | Same menu. `train()` is commented; it loads `checkpoints/glove_*.pt` and writes `glove_results.csv`. |

Copies of the recorded tables also live under `model/results/`.

## MOSEI — GMTM + ablation

| Script | Text | Does |
| --- | --- | --- |
| `model/train_GMTM_bert.py` | BERT | Identity encoders + GMTM fusion. Ablation list is currently only `['text', 'audio', 'visual']`; other combinations are commented. `train()` is commented; loads `checkpoints/ablation/model_{modalities}.pt`. |
| `model/train_GMTM_glove.py` | GloVe | Full 7-way modality list. `train()` commented. Writes `ablation_glove_results.csv`. |

Checkpoint name pattern:

```
model/checkpoints/ablation/model_{text+audio+visual}.pt
model/checkpoints/ablation/model_glove_{text+audio}.pt
```

## MOSI — transfer test (no training)

These load **MOSEI** checkpoints and score MOSI (train+valid+test merged).

| Script | Text | Output CSV |
| --- | --- | --- |
| `model/mosi_test/train_mosi_bert.py` | BERT | `mosi_bert_results.csv` |
| `model/mosi_test/train_mosi_glove.py` | GloVe | `mosi_glove_results.csv` (imports `training_structures.Supervised_Learning`, a leftover MultiBench path — prefer the BERT script’s `train_and_test` import if you revive this) |
| `model/mosi_test/mult_bert_mosi.py` | BERT GMTM | `ablation_mosi_results.csv` (write is commented) |
| `model/mosi_test/mult_glove_mosi.py` | GloVe GMTM | `ablation_mosi_glove_results.csv` |

## Shared library

| File | Role |
| --- | --- |
| `model/models.py` | All nn.Modules |
| `model/train_and_test.py` | `train`, `test`, `single_test`, `eval_affect`, bin helpers |
| `model/data/get_dataloader.py` | Pickle → DataLoader |
| `model/data/MOSEI/get_mosei.py` | SDK align + HDF5 dump (local paths) |
| `model/data/analysis_dataset.ipynb` | Peek at pickle shapes |
| `model/results/plot.ipynb` | Charts from the CSV tables |

## Examples added in this branch

| Script | Needs pickle? | Needs torch? |
| --- | --- | --- |
| `examples/summarize_recorded_results.py` | no (reads CSVs) | no |
| `examples/metrics_walkthrough.py` | no | no |
| `examples/fusion_forward_pass.py` | no | yes |
| `examples/gmtm_forward_pass.py` | no | yes |
| `examples/toy_train_loop.py` | no | yes |
| `examples/model_inventory.py` | no | yes |
