# Training

The only training loop is `train()` in `model/train_and_test.py`. Fusion
sweeps and GMTM are thin wrappers that build modules and call it.

## What `train()` does

```python
train(
    encoders, fusion, head,
    train_dataloader, valid_dataloader,
    total_epochs,
    additional_optimizing_modules=[],
    is_packed=False,
    early_stop=False,
    optimtype=torch.optim.RMSprop,   # scripts pass AdamW
    lr=0.001,                        # scripts pass 1e-4
    weight_decay=0.0,                # scripts pass 0.01
    objective=nn.CrossEntropyLoss(), # scripts pass L1Loss()
    save='best.pt',
    validtime=False,
    objective_args_dict=None,
    input_to_float=True,
    clip_val=8,
    track_complexity=True,
)
```

Per epoch:

1. `model.train()`, iterate the train loader with `tqdm`.
2. Build the input the way `is_packed` requires (see
   [datasets.md](datasets.md#two-collate-functions)).
3. `deal_with_objective(objective, pred, truth, args)` —
   for `L1Loss` / `MSELoss` this is just `objective(pred, truth.float())`.
4. Backward, `clip_grad_norm_(..., clip_val)`, optimizer step.
5. Full validation pass, mean loss over the valid set.
6. If valid loss improved, `torch.save(model, save)` (the **whole**
   `MultiFramework`, not a `state_dict`).
7. If `early_stop` and 8 straight non-improving epochs (`patience > 7`),
   break.

`track_complexity=True` (the default) wraps the run in
`memory_profiler.memory_usage` and prints wall time, peak RSS, and
parameter count. Install `memory-profiler` or pass
`track_complexity=False`. The example toy trainer does the latter by
calling GMTM directly instead of `train()`.

## Optimizer recipe used in this repo

Every published script that still has an active `train()` call
(`train_main_bert.py`, and the commented blocks in the others) uses:

| Knob | Value |
| --- | --- |
| `optimtype` | `torch.optim.AdamW` |
| `lr` | `1e-4` |
| `weight_decay` | `0.01` |
| `objective` | `torch.nn.L1Loss()` |
| `early_stop` | `True` |
| `clip_val` | 8 (function default) |
| batch size | 32 |
| `num_workers` | 0 |

`train_main_bert.py` currently passes `total_epochs=1`. That is almost
certainly a debug leftover; a real rerun should set this back to the
10–50 range used in the commented GMTM / GloVe calls
(`train_GMTM_bert.py` comments say 50, `train_GMTM_glove.py` says 20,
`train_main_glove.py` says 10). Validation early-stop is what actually
halted the published runs.

## Packed flag per method

| Fusion | `max_pad` loader | `is_packed` |
| --- | --- | --- |
| ConcatEarly, ConcatLate, TFN, LMF, TransformerLate | False (`_process_1`) | True |
| TransformerEarly | True (`_process_2`) | False |
| GMTM | True (`_process_2`) | False |

`train_main_bert.py` therefore builds **two** dataloader triples from the
same pickle (`traindata_OT` vs `traindata_TE`) and swaps them inside the
sweep.

## How to launch a sweep

From `model/`, with the pickle present:

```bash
python train_main_bert.py
```

The script writes `checkpoints/{FusionMethod}.pt` and a `main_results.csv`
in the current directory. Copy the CSV into `model/results/` if you want
the docs / plotter to pick it up (they read `model/results/` first).

GloVe and GMTM scripts have the `train()` call **commented** and instead
`torch.load` a previously saved `.pt`. Uncomment the block and point
`save=` at a new filename if you intend to retrain.

## MOSI scripts do not train

`model/mosi_test/train_mosi_bert.py` is named like a trainer but only
loads `../checkpoints/{method}.pt` and runs `test()` on the concatenated
MOSI loader. Same for `mult_bert_mosi.py` / `mult_glove_mosi.py`.

`train_mosi_glove.py` still imports
`training_structures.Supervised_Learning` (a MultiBench module). That
import will fail in a clean checkout of *this* repo. Use
`train_and_test.test` like the BERT sister script, or keep a MultiBench
clone on `PYTHONPATH` if you are reproducing an old environment.

## Checkpoint format

`torch.save(model, path)` pickles the entire `nn.Module`. Loading needs
the same class definitions (`models.MultiFramework` plus whatever fusion
class). A typical load in the scripts:

```python
model = torch.load(path).cuda()
```

That requires a GPU. To inspect a checkpoint on CPU:

```python
model = torch.load(path, map_location="cpu")
```

Do not `load_state_dict` into a freshly built GMTM unless you are sure
`embed_dim` / `n_features` match. The ablation checkpoints are one file
per modality subset (`model_text+audio+visual.pt`,
`model_glove_text.pt`, …) even though the architecture width is the same;
only the learned weights differ.

## Complexity helpers

`all_in_one_train` / `all_in_one_test` print:

- training wall time and peak memory
- parameter count via `getallparams`
- inference wall time

They are orthogonal to the metrics dictionary returned by `single_test`.

## Toy training without pickles

`examples/run_gmtm_toy.py` is the supported way to see a loss curve on a
machine that only has CPU torch:

- synthetic `(B, 50, F)` batches
- label = scaled mean of the text channel (so the task is learnable)
- `GatedMultiTransfomerModel` + `L1Loss` + `AdamW`
- asserts that the last epoch loss is lower than the first

That is a smoke test for the module, not a published score.
