# Training loop

All fitting goes through `train()` in
[`model/train_and_test.py`](../model/train_and_test.py). This page is a
guided reading of that function and of the script-level choices around it.

## Object graph

```
MultiFramework
├── encoders: ModuleList[n]
├── fuse: Fusion module          # Concat*, TensorFusion, GMTM, ...
└── head: nn.Module              # MLP / Identity / LSTM+MLP
```

`forward` optionally unpacks padded batches, stores `self.reps` (per-modality
encoder outputs) and `self.fuseout` (fusion tensor) for losses that want
intermediate activations, then returns `head(fused)`.

## Optimizer and objective

| Setting | Default in `train()` | Value used by the experiment scripts |
| --- | --- | --- |
| Optimizer | `RMSprop` | **`AdamW`** |
| `lr` | `1e-3` | **`1e-4`** |
| `weight_decay` | `0.0` | **`0.01`** |
| `objective` | `CrossEntropyLoss` | **`L1Loss`** |
| `clip_val` | `8` | `8` |
| `input_to_float` | `True` | `True` |
| `early_stop` | `False` | **`True`** (patience 7) |
| `track_complexity` | `True` | default (prints peak RAM + param count) |

`deal_with_objective` special-cases `CrossEntropyLoss` (squeeze + `.long()`),
`MSELoss` / `L1Loss` / `BCEWithLogitsLoss` (`.float()`), and otherwise calls
`objective(pred, truth, args)` so a custom loss can see `reps` / `fused`.

## One epoch

1. `model.train()`, tqdm over the train loader.
2. `op.zero_grad()`.
3. Packed path: `model([[x.float().to(device) for x in j[0]], j[1]])`.
4. Unpacked path: `model([x.float().to(device) for x in j[:-1]])`.
5. L1 against `j[-1]`.
6. `loss.backward()`, `clip_grad_norm_(parameters, 8)`, `op.step()`.
7. Switch to `eval()`, no-grad validation, mean L1.
8. If validation L1 improved, `torch.save(model, save)` (the **whole**
   `MultiFramework`, not a `state_dict`).
9. If `early_stop` and patience exceeds 7, break.

The saved file is later reloaded with `torch.load(path).cuda()` in the
scripts. That pattern needs the same `models.py` class definitions in
`sys.path` at load time.

## Script-level wiring

### Baseline sweep — `train_main_bert.py`

```text
epochs = 1          # currently a smoke-length run in the committed file
is_packed = False only for TransformerEarly, else True
save = checkpoints/{FusionMethod}.pt
```

BERT input dims `[35, 74, 768]`. Two loaders are built up front: padded
(`max_pad=True`) for early transformer, variable-length for everyone else.

`train_main_glove.py` is the same menu with dims `[35, 74, 300]` and
checkpoint names `checkpoints/glove_{FusionMethod}.pt`. The `train(...)`
call is commented out in the committed GloVe file (eval-only).

### GMTM — `train_GMTM_bert.py` / `train_GMTM_glove.py`

```text
encoders = Identity × 3
fusion   = GatedMultiTransfomerModel(3, input_dims, HParams)
head     = Identity
max_pad  = True
is_packed = False
objective = L1Loss
```

The BERT script's `train(...)` is commented out and only the three-way
modality list is enabled. The GloVe script iterates all 7 subsets. Ablation
checkpoints:

```
checkpoints/ablation/model_{text+audio+visual}.pt
checkpoints/ablation/model_glove_{text+audio}.pt
...
```

### MOSI scripts

`model/mosi_test/*.py` load those checkpoints and call `test()` only.
`get_mosi_dataloader` merges MOSI splits (see datasets note).

`train_mosi_glove.py` still imports
`training_structures.Supervised_Learning` — that is a leftover MultiBench
path and will fail unless that package is on `PYTHONPATH`. The other MOSI
files import `train_and_test` from this repo.

## Device

`train()` and `single_test()` pick `cuda:0` if `torch.cuda.is_available()`
else CPU. The experiment scripts additionally call `.cuda()` on every
submodule at construction time, so they assume a GPU when you uncomment
training.

The examples force CPU so they run in this environment without a GPU.

## What is *not* in the loop

- No learning-rate schedule (constant `1e-4`).
- No mixed precision.
- No class weighting (the task is regression).
- No unaligned / robustness loaders (`robust_test` is accepted by
  `get_dataloader` and ignored).
- Validation does not compute Acc-2 / F1; only L1. Classification metrics
  appear at test time.

## Tiny CPU stand-in

[`examples/tiny_train_loop.py`](../examples/tiny_train_loop.py) repeats the
same recipe on synthetic batches: `Identity` encoders, small GMTM, `L1Loss`,
`AdamW`, gradient clip 8, two short epochs. It exists to show the control
flow, not to reproduce the CSV numbers.
