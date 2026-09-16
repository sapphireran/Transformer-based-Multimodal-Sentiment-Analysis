# Experiments

Every original launch script is a personal research runner: hard-coded
paths, `.cuda()`, and CSV side effects. This page is a map of what each
file is supposed to do, including the commented-out training loops that
are currently left in "test only" mode.

Run them from `model/` unless noted.

## Shared training recipe

`train()` in `train_and_test.py`:

| Knob | Typical value in this repo |
| --- | --- |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Objective | `L1Loss` |
| Grad clip | 8 |
| Early stop | patience 7 on validation L1 |
| Batch size | 32 |
| Device | first CUDA device if the launch script called `.cuda()` |

`track_complexity=True` (the default) wraps the loop in `memory_profiler`,
so that dependency is required even for a one-epoch smoke run.

## MOSEI fusion sweep (BERT)

**Script:** `train_main_bert.py`

Loads `data/MOSEI/mosei_raw_bert.pkl` twice: variable-length (`max_pad=False`)
for most methods, padded (`max_pad=True`) for `TransformerEarly`.

| Method | Encoders | Fusion | Head | Packed? |
| --- | --- | --- | --- | --- |
| ConcatEarly | Identity ×3 | `ConcatEarly` | LSTM(877→1024) + MLP | yes |
| ConcatLate | LSTM 35/74/768 → 64/256/1024 | `ConcatLate` | MLP(1344) | yes |
| LowRankTensorFusion | GRUWithLinear → 32/64/256 | LMF rank 32 | MLP(256) | yes |
| TensorFusion | GRUWithLinear → 19/39/159 | outer product | MLP(128000→2048→1) | yes |
| TransformerEarly | Identity ×3 | `EarlyFusionTransformer(877)` | MLP(64) | no |
| TransformerLate | TransformerSeq 64/128/1024 | `LateFusionTransformer(1216)` | MLP(32) | yes |

The epoch count in the checked-in file is `total_epochs=1`. That looks like
a leftover from a debug pass; the numbers in `main_results.csv` are from an
earlier longer run (see [personal_lab_notes.md](personal_lab_notes.md)).
Checkpoints are written to `checkpoints/{FusionMethod}.pt`.

Results file: `main_results.csv` (also copied to `results/main_results.csv`).

## MOSEI fusion sweep (GloVe)

**Script:** `train_main_glove.py`

Same six methods, widths swapped to 300-d text:

| Method | Notable width change |
| --- | --- |
| ConcatEarly | LSTM on 409-d concat, hidden 512 |
| ConcatLate | text LSTM hidden 512, head 832 |
| LMF | GRU outs 32/64/128, fusion out 128 |
| TensorFusion | GRU outs 19/39/79, product 64000 |
| TransformerEarly | `n_features=409`, head is Identity |
| TransformerLate | TransformerSeq 64/128/512, `in_dim=1792` |

Training is commented out; the script loads `checkpoints/glove_{Method}.pt`
and writes `glove_results.csv`.

## GMTM on MOSEI (BERT)

**Script:** `train_GMTM_bert.py`

- Pickle: `data/MOSEI/mosei_raw_bert.pkl`
- Loader: `get_ablation_dataloader(..., embedding='bert', max_pad=True)`
- Model: `GatedMultiTransfomerModel(3, [35, 74, 768], HParams)`
- Encoders / head: `Identity`
- Active modality list: `['text', 'audio', 'visual']` (others commented)
- Train call commented; loads `checkpoints/ablation/model_text+audio+visual.pt`

CSV writing is also commented. The table that corresponds to a full BERT
ablation (including unimodal and bimodal rows) is `ablation_results.csv`.

## GMTM on MOSEI (GloVe)

**Script:** `train_GMTM_glove.py`

Same structure with `mosei_raw_glove.pkl`, `n_features=[35, 74, 300]`, and
all seven modality subsets enabled. Training is commented. Writes
`ablation_glove_results.csv` after loading
`checkpoints/ablation/model_glove_{text+audio+...}.pt`.

## MOSEI → MOSI transfer

Scripts under `model/mosi_test/` load **MOSEI-trained** checkpoints and
score them on MOSI pickles. They do not train.

| Script | Checkpoint glob | MOSI pickle | Output CSV |
| --- | --- | --- | --- |
| `train_mosi_bert.py` | `../checkpoints/{Method}.pt` | `mosi_raw_bert.pkl` | `mosi_bert_results.csv` |
| `train_mosi_glove.py` | `../checkpoints/glove_{Method}.pt` | `mosi_raw_glove.pkl` | `mosi_glove_results.csv` |
| `mult_bert_mosi.py` | `../checkpoints/ablation/model_….pt` | BERT MOSI | (write commented) |
| `mult_glove_mosi.py` | `../checkpoints/ablation/model_glove_….pt` | GloVe MOSI | `ablation_mosi_glove_results.csv` |

`train_mosi_glove.py` still imports
`training_structures.Supervised_Learning` (a MultiBench leftover) instead of
the local `train_and_test`. That import will fail in this repo as checked
in.

MOSI loaders merge train+valid+test. Do not treat those CSVs as official
split scores.

`train_mosi_bert.py` names late concat `Concat` in the method list, and
looks for `checkpoints/Concat.pt`. The MOSEI sweep saves `ConcatLate.pt`.
If you re-run transfer, either rename the checkpoint or the method string.

## Notebooks

| Notebook | Role |
| --- | --- |
| `model/data/analysis_dataset.ipynb` | Inspect pickle keys, plot label histograms |
| `model/data/MOSI/get_dataset.ipynb` | SDK download / align session (Windows paths) |
| `model/data/MOSEI/get_dataset.ipynb` | same for MOSEI |
| `model/results/plot.ipynb` | Chart the CSV tables |

## Synthetic stand-ins (no dataset required)

Use these when you want to verify modules without CUDA or pickles:

| Script | What it proves |
| --- | --- |
| `examples/forward_pass_demo.py` | GMTM and classical fusions produce finite `[B, 1]` (or known ranks) |
| `examples/fusion_shape_walkthrough.py` | prints the rank contract of each block |
| `examples/toy_train_gmtm.py` | GMTM can overfit a synthetic linear mixture |
| `examples/tiny_affect_dataset.py` | pickle schema + a minimal Dataset |
| `examples/summarize_results.py` | reprints every CSV as ranked tables |
| `examples/metric_walkthrough.py` | Acc7 edges and hand-checked metric cases |
