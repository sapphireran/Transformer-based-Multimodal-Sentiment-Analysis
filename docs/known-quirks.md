# Known quirks in the original upload

These are properties of the personal research snapshot, not of the
examples lab. The lab works around them rather than rewriting every
training script.

## Name collision: `Linear`

`model/models.py` defines `class Linear(torch.nn.Module)` around line 201
and later `def Linear(in_features, out_features, bias=True)` around line
360. The function wins. `TransformerEncoderLayer` uses the function as a
factory, which is fine. Anything that expected the class will get a
function instead.

## `EarlyFusionTransformer` `batch_first`

The encoder layer is constructed with `batch_first=True`, then `forward`
permutes to `[seq, batch, embed]` before calling the encoder. Behaviour
depends on the PyTorch version's `batch_first` interpretation. Do not
treat this class as a reference implementation of "batch first
transformers"; GMTM uses a custom encoder instead.

## `test()` signature drift

Local `train_and_test.test` is:

```text
test(model, test_dataloaders_all, is_packed=False, criterion=..., input_to_float=True)
```

Several GloVe / MOSI scripts still pass MultiBench kwargs (`dataset=`,
`no_robust=`). Those calls `TypeError` on this tree. BERT MOSEI
`train_main_bert.py` uses the local signature and is the better template.

## Missing MultiBench import

`model/mosi_test/train_mosi_glove.py` starts with

```text
from training_structures.Supervised_Learning import train, test
```

That package is not vendored. Use `train_and_test` like the BERT MOSI
file, or vendor MultiBench.

## Epoch count vs CSV tables

`train_main_bert.py` passes `total_epochs=1`. The numbers in
`model/results/main_results.csv` are from a longer personal run. Re-running
the file as-is will not reproduce the table.

## GMTM scripts do not train as committed

`train_GMTM_bert.py` and `train_GMTM_glove.py` comment out `train(...)`
and only `torch.load` a checkpoint. Ablation combination lists are also
partially commented (BERT currently keeps only the trimodal tuple).

## Headless `plt.show()`

`single_test` always opens a confusion-matrix window. SSH / CI / this
cloud agent will hang or error. `model/metrics.py` is the headless path.

## `memory_profiler` import

`train_and_test.py` imports `memory_usage` at module level, so merely
importing `train` requires `memory-profiler` even when
`track_complexity=False`. It is listed in `requirements.txt`. The
examples lab does not import `train_and_test`.

## Low-rank ones vector device

`LowRankTensorFusion.forward` builds `ones` on `cuda:0` if any GPU is
visible. Mixing CPU tensors with that vector will throw. The examples
force CPU.

## MOSI loader merges splits

`get_mosi_dataloader` concatenates train, valid, and test. MOSI CSVs are
transfer dumps, not official MOSI test-set scores.

## `get_mosei.py` naming

The file lives under `model/data/MOSEI/` but the constants are
`MOSI_PATH`, `md.cmu_mosi`, and `CMU_MOSI_*` fields. Treat it as a MOSI
SDK sketch that was copied into the MOSEI folder.

## GloVe late-transformer `in_dim`

`train_main_glove.py` builds `TransformerSeq` widths 64+128+512 = 704
but constructs `LateFusionTransformer(in_dim=1792)`. A fresh GloVe late
transformer run needs that integer reconciled with the actual cat width.

## `MLP` dropout on the wrong tensor

`MLP.forward` applies the second dropout to `output` (the hidden
activation) instead of `output2`. Existing checkpoints were trained with
that behaviour; do not "fix" it if you want to load those weights.

## Unused GMTM residual

`self.alpha` is a parameter. The residual add that would use it is
commented out in `forward`. It still appears in the optimiser's
parameter list.
