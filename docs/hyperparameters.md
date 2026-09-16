# Hyperparameters used in the committed CSVs

Change these and you are no longer comparable to `model/*.csv`. The values
below are copied from the training scripts, not from a config file — this
repo does not have one.

## Shared optimisation

| Knob | Value | Where |
| --- | --- | --- |
| loss | `L1Loss` (MAE) | every `train(...)` call |
| optimizer | AdamW | `train_main_*.py`, `train_GMTM_*.py` |
| learning rate | `1e-4` | same |
| weight decay | `0.01` | same |
| grad clip | `8` | `train_and_test.train` default |
| batch size | 32 | dataloaders |
| `max_seq_len` | 50 | dataloaders |
| early stop patience | 7 epochs | `early_stop=True` |
| device | `cuda:0` | `.cuda()` at construction |

`train_main_bert.py` currently passes `total_epochs=1` (a smoke-length
sweep). The GloVe / GMTM scripts have the `train(...)` call commented and
load checkpoints instead. Uncomment and raise epochs for a real fit.

## GMTM (`HParams` in `train_GMTM_bert.py` / `train_GMTM_glove.py`)

| Field | Value |
| --- | --- |
| `embed_dim` | 64 |
| `num_heads` | 4 (must divide `embed_dim`) |
| `layers` | 4 |
| `attn_dropout` | 0.1 |
| `attn_dropout_modalities` | `[0, 0, 0.1]` (vision, audio, text) |
| `relu_dropout` / `res_dropout` | 0.1 |
| `embed_dropout` | 0.2 |
| `out_dropout` | 0.1 |
| `output_dim` | 1 |
| `attn_mask` | `True` |

The CPU examples use `examples/gmtm_forward.py:TinyHParams`
(`embed_dim=16`, 1 layer, 2 heads, no dropout) so a laptop can finish a
forward and a short train. `--preset paper` rebuilds the table above.

## Encoder widths

See the tables in [training.md](training.md). Short version:

* BERT text = 768, GloVe text = 300, vision = 35, audio = 74
* LMF rank = 32 in both views
* TFN encoder outputs are truncated so the Kronecker product fits in
  memory (128000-d BERT head, 64000-d GloVe head)

## Ablation protocol (do not “remove” modules)

Keep three GMTM inputs. Zero the dropped modality with the shapes in
`get_ablation_dataloader`. Parameter count stays constant; only the
information in that stream changes. The seven subsets are listed in
`model/checkpoints/ablation/readme.md`.

## What the examples deliberately change

| Example | Divergence |
| --- | --- |
| `TinyHParams` | smaller graph, CPU, no dropout |
| `train_toy_gmtm.py` | `lr=1e-3`, tens of synthetic clips, 6–8 epochs |
| toy labels | generated from a planted cue, not MOSI annotators |

Those runs validate the **code path**, not the CSV numbers.
