# Known quirks in the original scripts

Personal errata. None of these are “bugs to silently fix in a drive-by
commit” — they change how you read the CSVs and how you rerun things.

## Naming

- `GatedMultiTransfomerModel` is missing an `r`. Imports must use the
  misspelled name.
- MOSI fusion tables call late concat `Concat` instead of `ConcatLate`.
  Same module (`ConcatLate`).

## `models.py`

- `class Linear` is later **shadowed** by `def Linear(...)`. Anything after
  that line uses the factory function (Xavier uniform). The class is dead
  code.
- `EarlyFusionTransformer` sets `batch_first=True` then permutes the tensor
  to `[T, B, C]` before `nn.TransformerEncoder`. PyTorch then treats the
  **time** dimension as the batch. The module still runs; the attention
  pattern is not the “batch-first” one the constructor claims.
- `GatedMultiTransfomerModel.trans_mems` is built and never called (the
  line is commented). Those weights sit in the checkpoint unused.
- `self.alpha` is an unused residual knob.
- `LowRankTensorFusion` hardcodes
  `cuda:0 if torch.cuda.is_available() else cpu` for the ones vector
  instead of `modality.device`.
- `TransformerEncoderLayer.apply_sublayer` does
  `norm(x + dropout(x))` *after* attention already wrote `x`. Residual
  bookkeeping is not the standard Pre-LN MulT residual. Treat the custom
  encoder as “a transformer-like stack,” not a line-for-line MulT port.
- `create_attention_mask` always returns `None`, so `attn_mask=True` in
  `HParams` does not actually mask future positions.

## Training scripts vs. `test()` signature

Current `test()`:

```python
def test(model, test_dataloaders_all, is_packed=False,
         criterion=nn.CrossEntropyLoss(), input_to_float=True):
```

Several scripts still call:

```python
test(..., dataset='mosei', no_robust=True)
```

That matches an older MultiBench-style signature. Those extra kwargs will
`TypeError` on the `train_and_test.py` that is in this tree. Affected:

- `train_GMTM_glove.py`
- `train_main_glove.py`
- `mosi_test/train_mosi_glove.py`
- `mosi_test/train_mosi_bert.py` (`dataset='mosi'` only)

If you uncomment those test calls, drop the unknown kwargs.

## MOSI GloVe script imports a missing package

```python
from training_structures.Supervised_Learning import train, test
```

That module is not in this repository. `train_mosi_glove.py` will not
import until it is pointed at `train_and_test` like the BERT sibling.

## GloVe late-transformer width

`train_main_glove.py` builds

```
TransformerSeq(35, 64) + TransformerSeq(74, 128) + TransformerSeq(300, 512)
```

concat width `704`, but `LateFusionTransformer(in_dim=1792)`. A fresh
forward will fail or silently mis-project depending on how you concatenate.
The GloVe TransformerLate CSV row should be treated as “checkpoint from
the original box,” not as something you can reconstruct from the script
body alone.

BERT late path *is* consistent: `64+128+1024 = 1216`.

## Data

- `get_mosei.py` lives under `MOSEI/` but loads `cmu_mosi` constants and
  `CMU_MOSI_*` field names. Copy-paste leftover.
- Absolute paths (`F:\MOSEI\...`, `/home/van/backup/pack/mosi/mosi.hdf5`)
  are from the original machines and are unused once you pass a pickle
  filepath.
- `get_mosi_dataloader` concatenates train+valid+test. Transfer CSVs are
  not official MOSI test scores.
- `torchtext` is imported in `get_dataloader.py` and unused.

## Evaluation UX

- `single_test` calls `plt.show()` on a 7-way confusion matrix. Headless
  or CI runs should patch that out (the examples never call it).
- `import train_and_test` requires `memory_profiler` at import time.
- Checkpoints are full-module pickles (`torch.save(model, path)`). They
  are brittle across PyTorch versions.

## Epoch counts in the committed scripts

`train_main_bert.py` asks for **1** epoch. GMTM files have 20–50 epoch
calls commented out. The CSVs are from longer personal runs that are not
fully reflected in the current `__main__` blocks.

## Ablation is zero-masking, not architecture search

“Audio-only GMTM” is still a 3-tower model with two zero inputs. Parameter
count does not drop. A fairer “unimodal” run would instantiate
`n_modalities=1`. The tables are still useful for “what happens if this
stream is missing,” which is what the zero mask actually tests.
