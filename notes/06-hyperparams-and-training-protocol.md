# 06 — Hyperparameters and training protocol

Reconstructed from the scripts as they sit in git, plus what the CSVs
imply. Where the script and the CSV cannot both be true, I say so.

## Shared defaults

| Knob | Value | Where |
| ---- | ----- | ----- |
| Optimizer | AdamW | all train_* scripts |
| lr | 1e-4 | all |
| weight_decay | 0.01 | all |
| loss | L1 (`nn.L1Loss`) | all |
| grad clip | 8 | `train_and_test.train` default `clip_val` |
| batch size | 32 | all |
| max sequence | 50 | dataloaders |
| early stop | patience 7 on valid L1 | `early_stop=True` |
| device | `cuda` required | `.cuda()` in constructors |
| seed | **unset** | — |
| checkpoint | `torch.save(model, path)` full module | `train()` |

`train()` also wraps the loop in `memory_profiler.memory_usage` when
`track_complexity=True` (the default). That is why a train job prints
peak RSS.

## Per-script epoch counts (as committed)

| Script | `total_epochs` argument | Train call live? |
| ------ | ----------------------: | ---------------- |
| `train_main_bert.py` | **1** | yes |
| `train_main_glove.py` | 10 (commented) | **no** — load + test only |
| `train_GMTM_bert.py` | 50 (commented) | **no** |
| `train_GMTM_glove.py` | 20 (commented) | **no** |
| `mosi_test/*` | n/a | test only |

I do **not** believe the BERT fusion CSV is a 1-epoch table. Acc-2 0.84
and Corr 0.70 after one pass of MOSEI would be surprising for ConcatEarly,
and the relative ranking is too tidy. More likely I dropped the epoch
count when I last used the file as a smoke test and committed that.

GMTM comments (50 / 20) match how I remember those runs: longer on BERT
because it was the “real” model, shorter on GloVe because I was impatient.
I cannot prove it from git.

## Packed vs max-pad (this changes the optimizer trajectory)

| Method | Loader | `is_packed` |
| ------ | ------ | ----------- |
| ConcatEarly / Late, LMF, TFN, TransformerLate | packed, variable length | True |
| TransformerEarly | max-pad 50 | False |
| GMTM (all subsets) | max-pad 50 | False |

So the bake-off is not “six fusions, one data pipeline.” It is two
pipelines. I keep forgetting this when I stare at TransformerEarly.

## Encoder / fusion widths (copy-paste checklist)

BERT (`input_dims = [35, 74, 768]`):

| Method | Encoders | Fusion / head |
| ------ | -------- | ------------- |
| ConcatEarly | 3× Identity | concat 877 → LSTM 1024 → MLP 1024→1 |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | cat 1344 → MLP 1344→1 |
| LMF | GRU+lin 35→64→32, 74→256→64, 768→1024→256 | LMF rank 32 → 256 → MLP 256→1 |
| TFN | GRU+lin → 19 / 39 / 159 | TFN → MLP 128000→2048→1 |
| TransformerEarly | 3× Identity | EarlyTF 877→32, 4×4 → MLP 64→1 |
| TransformerLate | TFSeq 35→64, 74→128, 768→1024 | LateTF in_dim=1216 → 32 → MLP 32→1 |
| GMTM | 3× Identity | GMTM 64-d, 4 layers, 4 heads → 1 |

GloVe (`[35, 74, 300]`):

| Method | Encoders | Fusion / head |
| ------ | -------- | ------------- |
| ConcatEarly | 3× Identity | concat 409 → LSTM 512 → MLP 512→1 |
| ConcatLate | LSTM 35→64, 74→256, 300→512 | cat 832 → MLP 832→1 |
| LMF | GRU+lin → 32 / 64 / 128 | LMF rank 32 → 128 → MLP 128→1 |
| TFN | GRU+lin → 19 / 39 / 79 | TFN → MLP 64000→2048→1 |
| TransformerEarly | 3× Identity | EarlyTF 409→32, head **Identity** |
| TransformerLate | TFSeq 35→64, 74→128, 300→512 | LateTF **in_dim=1792** (likely wrong; 64+128+512=704) |
| GMTM | 3× Identity | same HParams as BERT, text Linear 300→64 |

I narrowed the GloVe LSTMs/GRUs on purpose (1024→512 on text) so the
weaker front did not carry BERT-sized heads.

## GMTM HParams (both fronts)

```text
num_heads = 4
layers = 4
attn_dropout = 0.1
attn_dropout_modalities = [0, 0, 0.1]
relu_dropout = 0.1
res_dropout = 0.1
out_dropout = 0.1
embed_dropout = 0.2
embed_dim = 64
attn_mask = True          # LN on; mask is None
output_dim = 1
all_steps = False         # unused
modality_dropout = 0.2    # unused
use_text_transformer = True  # unused
```

## Validation and model selection

`train()` keeps the checkpoint with the **lowest valid L1**. It does not
look at Acc-2 or Corr. A run that is great on polarity and worse on MAE
will lose to a slightly better regressor.

`pts` is appended in the train loop and never used. Leftover from an
older “print predictions” debug.

## Test-time extras

`single_test` always:

- computes MSE (not stored)
- builds a 7-way confusion matrix
- calls `plt.show()`

Headless: set `MPLBACKEND=Agg` or the process waits for a display.

## What I would standardize before the next sweep

```text
epochs = 40
patience = 8
seeds = [13, 37, 101]
lr = 1e-4
weight_decay = 0.01
batch = 32
loss = L1
clip = 8
save = state_dict + json(hparams, seed, git_sha, valid_mae)
log = csv row per epoch (train_l1, valid_l1, valid_mae, valid_acc2)
```

And one loader per comparison. If I want packed vs max-pad as a factor,
it becomes its own column, not an implicit property of TransformerEarly.

## Personal takeaway

The protocol is “AdamW + L1 + valid MAE,” which I still like for this
label. The protocol is also “one seed, two loaders, scripts that do not
match the CSVs.” I trust the **ranking inside a CSV** more than I trust
any absolute float, and I trust a CSV more than I trust the current
epoch arguments.
