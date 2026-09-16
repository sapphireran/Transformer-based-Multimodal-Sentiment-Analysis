# Training loop

The shared trainer is `train()` / `test()` / `single_test()` in
[`model/train_and_test.py`](../model/train_and_test.py).

## `MultiFramework`

```python
class MultiFramework(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False): ...
    def forward(self, inputs):
        outs = [enc(x) for enc, x in zip(self.encoders, inputs)]
        fused = self.fusion(outs)
        return self.head(fused)
```

When `has_padding=True` (packed LSTM/GRU paths), each encoder receives
`[tensor, lengths]` and CuDNN is disabled for that step so packed sequences
backward cleanly.

GMTM construction used in the ablation scripts:

```python
encoders = [Identity(), Identity(), Identity()]
fusion   = GatedMultiTransfomerModel(3, [35, 74, 768], hyp_params=HParams)
head     = Identity()
```

## Default optimization (main BERT sweep)

From [`train_main_bert.py`](../model/train_main_bert.py):

| Knob | Value |
| --- | --- |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Objective | `nn.L1Loss()` (MAE on the raw score) |
| Grad clip | `8.0` (`clip_grad_norm_`) |
| Early stop | validation L1, patience **7** |
| Batch size | 32 |
| Device | first CUDA device if present, else CPU |
| Checkpoint | `torch.save(model, path)` — full module, not `state_dict` |

`train_main_bert.py` currently calls `train(..., total_epochs=1)` which is a
smoke setting. The numbers in `model/results/*.csv` come from earlier,
longer runs (GMTM comments mention 20–50 epochs). Do not expect a 1-epoch
rerun to match the tables.

`train_main_glove.py` and the GMTM scripts have the `train(...)` call
**commented out** and only `torch.load` a checkpoint. Re-training those
requires uncommenting the block and pointing `save=` at a writable path.

## Packed vs max-pad

Two collate functions exist in [`get_dataloader.py`](../model/data/get_dataloader.py):

| Flag | Collate | Batch layout | Used by |
| --- | --- | --- | --- |
| `max_pad=False` | `_process_1` | `(list_of_padded_mods, lengths, ids, y)` | Concat / TFN / LMF / TransformerLate (`is_packed=True`) |
| `max_pad=True` | `_process_2` | `(vision, audio, text, y)` all `[B, 50, F]` | TransformerEarly, GMTM (`is_packed=False`) |

`_process_1` pads each modality independently to the batch-max length.
`_process_2` stacks the already-truncated `[50, F]` clips.

`TransformerEarly` is the only main-sweep method that forces `max_pad=True`
and `is_packed=False`. Mixing the two flags will crash on a shape or on
`pack_padded_sequence`.

## Loss adapter

`deal_with_objective` reshapes targets for `CrossEntropyLoss` vs. regression.
All sentiment runs in this repo use `L1Loss` or theoretically `MSELoss` /
`BCEWithLogitsLoss`. The CE branch is leftover from a classification sketch.

## Complexity tracker

If `track_complexity=True` (the default), `train()` wraps the loop in
`memory_profiler.memory_usage` and prints peak RSS plus parameter count.
That import is **module-level**, so `import train_and_test` requires
`memory-profiler` even if you only want `eval_affect`.

The examples avoid that import and ship their own metric helpers.

## Checkpoint format

`torch.save(model, "checkpoints/ConcatLate.pt")` pickles the entire
`MultiFramework` instance. Loading needs the same class definitions in
`models.py` / `train_and_test.py`. There is no `state_dict` export in the
training scripts.

Git only has placeholder readmes under `model/checkpoints/`. The `.pt`
files from the original Windows machine were never committed (and should
not be — they are large and environment-specific).

## Suggested personal rerun (BERT GMTM)

```bash
cd model
# uncomment train(...) in train_GMTM_bert.py first
python train_GMTM_bert.py
```

Expect: one pass per modality subset listed in `modality_combinations`,
L1 on MOSEI valid, write
`checkpoints/ablation/model_text+audio+visual.pt` (name is `'+'.join`).
Then `test()` prints MAE / Acc / Corr / F1 and pops a confusion-matrix
window unless you stub `plt.show`.
