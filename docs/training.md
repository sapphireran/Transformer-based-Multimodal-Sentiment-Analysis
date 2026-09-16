# Training

The supervised loop lives in `model/train_and_test.py` (`train`, `test`, `single_test`). Experiment scripts only choose encoders, fusion, head, pickle path, and hyperparameters.

## Objective

Recorded runs use **`torch.nn.L1Loss`** on a scalar prediction vs. the continuous sentiment label. `deal_with_objective` also knows CrossEntropy, MSE, and BCE-with-logits, but those are unused in the checked-in scripts.

Gradient clipping is `clip_val=8` (global L2 norm).

## Optimizer

```
AdamW, lr=1e-4, weight_decay=0.01
```

`train_main_bert.py` currently passes `total_epochs=1` (a short smoke / leftover). GMTM comments show `50` epochs (BERT) and `20` epochs (GloVe) with `early_stop=True`. Early stop fires after **8** epochs without a new best validation L1 (`patience > 7`).

Best checkpoint is the full `nn.Module` (`torch.save(model, path)`), not a `state_dict`. Loading needs the same class definitions on `sys.path`.

## Device

Scripts call `.cuda()` on modules and `torch.load(...).cuda()`. There is no CPU/DataParallel switch in the original files. Synthetic examples in `examples/` use CPU so they run on this cloud checkout.

## GMTM hyperparameters

Shared by `train_GMTM_bert.py`, `train_GMTM_glove.py`, and the MOSI GMTM testers:

```python
class HParams:
    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 64
    attn_mask = True
    output_dim = 1
    all_steps = False
    modality_dropout = 0.2          # stored; not read inside GMTM.forward
    use_text_transformer = True     # stored; not read inside GMTM.forward
```

`modality_dropout` and `use_text_transformer` are unused in the module. Dropout that actually runs is `embed_dropout` on the input projections, the custom encoder dropouts, and `out_dropout` on the head.

## Batching

- `batch_size = 32`
- `max_seq_len = 50`
- `num_workers = 0` in the scripts (avoids Windows / notebook pickle issues)
- `data_type = 'mosei'` or `'mosi'` only affects `_get_class` (unused when labels stay continuous)

## Checkpoint locations

| Experiment | Path pattern |
| --- | --- |
| BERT bake-off | `model/checkpoints/{FusionMethod}.pt` |
| GloVe bake-off | `model/checkpoints/glove_{FusionMethod}.pt` |
| GMTM BERT ablation | `model/checkpoints/ablation/model_{text+audio+visual}.pt` |
| GMTM GloVe ablation | `model/checkpoints/ablation/model_glove_{...}.pt` |

The `readme.md` files under `checkpoints/` are only location markers. Weights are not in git.

## Train vs. test only

Several scripts have `train(...)` commented and only run `test(...)`. That matches a workflow of “train once on a GPU box, commit CSVs, re-score later”. To train from scratch, uncomment the `train(` block and make sure the pickle path exists.

`train_main_bert.py` is the one bake-off script that still calls `train` in the committed source (for 1 epoch).

## Complexity tracking

Default `track_complexity=True` imports `memory_profiler` and times the whole `train` closure. Install it (`requirements.txt`) or pass `track_complexity=False`.

## Minimal mental model of one step

```python
out = model([vision.float(), audio.float(), text.float()])  # or packed variant
loss = L1(out, label.float())
loss.backward()
clip_grad_norm_(model.parameters(), 8)
optimizer.step()
```

`examples/toy_train_loop.py` is that loop on synthetic batches with a small GMTM (2 layers, embed 16) so you can watch train L1 drop without MOSI/MOSEI.
